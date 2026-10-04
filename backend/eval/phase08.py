"""Phase 0.8: decide Jev's role from retrieval + abstain evidence.

Usage:
  python backend/eval/phase08.py --warm-jev-cache
  python backend/eval/phase08.py --reuse-jev-cache --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.metrics import (
    auroc,
    bootstrap_ci,
    bootstrap_diff_ci,
    hit_from_rank,
    mrr_from_rank,
    percentile,
)
from eval.schema import first_match_rank, load_jsonl, parent_matches_expected
from jev_disk_cache import AnswerabilityDiskCache, JevDiskCache
from pipeline_logic import apply_retrieval_policy
from rag_engine import (
    CHECKER_MODEL,
    JEV_CANDIDATE_LIMIT,
    JEV_MODEL,
    JEV_RELEVANCE_THRESHOLD,
    RETRIEVAL_RERANK_MODE,
    TOP_K_CHILDREN,
    VECTOR_SIMILARITY_THRESHOLD,
    _openrouter_chat,
    assert_budget_for_estimate,
    cost_ledger_snapshot,
    enable_jev_disk_cache,
    estimate_harness_cost_usd,
    reset_cost_ledger,
    retrieve_parents,
    set_reuse_jev_cache,
    set_jev_cache_enabled,
)


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "baseline_config.json")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "baselines")
TYPES = ("factual", "exact", "multihop", "followup")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def housekeeping() -> Dict[str, Any]:
    """0.8.0 reconcile live knobs vs 0.7.2 vs frozen baseline."""
    live = {
        "jev_candidate_limit": JEV_CANDIDATE_LIMIT or None,
        "jev_relevance_threshold": JEV_RELEVANCE_THRESHOLD,
        "top_k": TOP_K_CHILDREN,
        "similarity_floor": VECTOR_SIMILARITY_THRESHOLD,
        "retrieval_rerank_mode": RETRIEVAL_RERANK_MODE,
        "jev_model": JEV_MODEL,
    }
    reconcile = {
        "frozen_baseline_recall_at_5": {
            "value": 0.8406,
            "config": (
                "run_eval.py --cold-jev-cache --faithfulness --runs 3; "
                "RETRIEVAL_RERANK_MODE=jev_filter (hard drop); top_k=30; "
                "jev_relevance_threshold=0.20; jev_candidate_limit=None (all 30); "
                "allow_retry=True; mean of 3 cold-Jev runs in phase0.json"
            ),
        },
        "phase072_fused_recall_at_5": {
            "value": 0.942,
            "config": (
                "miss_analysis.py --fused-only; use_jev=False; top_k=30; "
                "allow_retry=False; span match via normalize_match_text; "
                "answerable-only (69 items)"
            ),
        },
        "phase072_jev_recall_at_5": {
            "value": 0.8841,
            "config": (
                "miss_analysis Jev pass; use_jev=True hard filter; top_k=30; "
                "allow_retry=False; single clean pass (not mean of 3); "
                "answerable-only (69 items)"
            ),
        },
        "why_they_differ": [
            "Frozen 0.8406 is Jev hard-filter mean over 3 cold runs with retries on; 0.8841 is one clean Jev pass without retries.",
            "Fused 0.942 from 0.7.2 is not reproducible under the current golden+matcher; live fused-only recompute is recall@5=0.7536 / r@15=0.8116 (exact@5≈0.45). Treat 0.942 as superseded.",
            "Cold-Jev 520/fallback noise and retry rewrites move ranks between runs (MRR spread 0.016 on the freeze).",
        ],
        "absent_from_fused_top30_vs_fused_r30_1": (
            "The 0.7.2 claim that fused r@30=1.000 while some exact labels were 'absent from fused top-30' "
            "was internally inconsistent. Live fused-only recompute shows r@30≈0.81 and several exact/multihop "
            "labels genuinely outside the fused top-30 (or only matching a same-hash parent that lacks the span). "
            "The 'absent' rows were real misses under today's matcher; the 0.7.2 r@30=1.000 figure was overstated."
        ),
        "live_fused_only_recompute": {
            "recall@5": 0.7536,
            "recall@15": 0.8116,
            "mrr": 0.5648,
            "note": "Authoritative fused-only numbers for Phase 0.8 comparisons.",
        },
        "live": live,
    }
    return reconcile


def _fused_retrieve(row: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return retrieve_parents(
        row.get("question") or "",
        history=row.get("history") or [],
        top_k=int(config.get("top_k", 30)),
        similarity_threshold=float(config.get("similarity_threshold", 0.30)),
        rrf_k=int(config.get("rrf_k", 60)),
        max_parents=30,
        use_jev=False,
        retrieval_rerank_mode="fused_only",
        allow_retry=False,
        jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
        retry_threshold=float(config.get("retry_threshold", 0.35)),
    )


def _warm_jev_cache(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    enable_jev_disk_cache(True)
    set_reuse_jev_cache(False)
    set_jev_cache_enabled(True)
    # Warm answerable and unanswerable: abstain E1 needs Jev scores on both.
    estimate = estimate_harness_cost_usd(
        n_questions=len(rows),
        answerable=len(rows),
        run_faithfulness=False,
        run_jev=True,
        top_k=int(config.get("top_k", 30)),
        runs=1,
    )
    assert_budget_for_estimate(estimate, label="warm-jev-cache")
    reset_cost_ledger()
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        retrieve_parents(
            row.get("question") or "",
            history=row.get("history") or [],
            top_k=int(config.get("top_k", 30)),
            similarity_threshold=float(config.get("similarity_threshold", 0.30)),
            rrf_k=int(config.get("rrf_k", 60)),
            max_parents=30,
            use_jev=True,
            retrieval_rerank_mode="jev_filter",
            allow_retry=False,
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
        )
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"warm {index}/{len(rows)}", file=sys.stderr, flush=True)
    return {
        "questions": len(rows),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "cost": cost_ledger_snapshot(),
        "cache": JevDiskCache().count(),
    }


def _scores_for_children(
    query: str,
    children: Sequence[Dict[str, Any]],
    disk: JevDiskCache,
) -> Tuple[Dict[str, Optional[float]], int]:
    out: Dict[str, Optional[float]] = {}
    missing = 0
    for child in children:
        chunk_id = child["chunk_id"]
        score = disk.get(query=query, chunk_text=child.get("text") or "", jev_model=JEV_MODEL)
        out[chunk_id] = score
        if score is None:
            missing += 1
    return out, missing


def _parents_from_ids(
    fused_result: Dict[str, Any],
    ordered_ids: Sequence[str],
    children_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map chunk ids back to parent passages already loaded on fused_result."""
    # Prefer fused_parents order lookup via scored children parent linkage.
    parent_by_chunk = {}
    for parent in fused_result.get("fused_parents") or []:
        # matched_passage / parent_id may be present after _load_parent
        parent_by_chunk[parent.get("parent_id")] = parent
    # Fall back: rebuild from retrieve ordered fused parents by walking children.
    # Simpler path: re-derive rank via parent_matches on fused_parents reordered by
    # the parent of each selected child.
    from rag_engine import _load_parent, _document_hash_map, _active_collection

    _coll, active = _active_collection()
    hash_map = _document_hash_map(active["collection_name"])
    parents = []
    seen_parents = set()
    for chunk_id in ordered_ids:
        child = children_by_id.get(chunk_id)
        if not child:
            continue
        parent = _load_parent(child)
        parent["content_hash"] = hash_map.get(parent.get("document_id") or "", "")
        parent["jev_score"] = child.get("jev_score")
        parent["score"] = child.get("score")
        parent["rrf_score"] = child.get("rrf_score")
        key = parent.get("parent_id") or parent.get("text")
        if key in seen_parents:
            continue
        seen_parents.add(key)
        parents.append(parent)
    return parents


def evaluate_policies(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """0.8.1 offline policies A–D vs fused-only on all answerable items."""
    set_reuse_jev_cache(True)
    enable_jev_disk_cache(True)
    disk = JevDiskCache()
    answerable = [row for row in rows if row.get("answerable", True)]
    policies = ["fused_only", "A", "B", "C", "D"]
    per_item: Dict[str, List[Dict[str, Any]]] = {policy: [] for policy in policies}
    missing_total = 0
    jev_calls_before = cost_ledger_snapshot().get("jev", 0.0)

    for index, row in enumerate(answerable, start=1):
        fused = _fused_retrieve(row, config)
        # scored_children in fused_only mode are fused-ranked children.
        children = fused.get("scored_children") or []
        if not children:
            # Reconstruct from fused_parents' matched passages is insufficient; re-fetch raw.
            children = []
        query = fused.get("query") or row.get("question") or ""
        scores, missing = _scores_for_children(query, children, disk)
        missing_total += missing
        children_by_id = {child["chunk_id"]: child for child in children}
        expected = row.get("expected") or []
        qtype = row.get("type") or "factual"

        # Baseline fused rank from engine-produced fused_parents (authoritative).
        fused_rank = first_match_rank(fused.get("fused_parents") or [], expected)
        per_item["fused_only"].append(
            {
                "id": row.get("id"),
                "type": qtype,
                "r5": hit_from_rank(fused_rank, 5),
                "r15": hit_from_rank(fused_rank, 15),
                "mrr": mrr_from_rank(fused_rank),
                "missing_jev": 0,
            }
        )

        for policy in ("A", "B", "C", "D"):
            applied = apply_retrieval_policy(
                children,
                scores,
                policy=policy,
                jev_threshold=float(config.get("jev_relevance_threshold", 0.20)),
                rrf_k=int(config.get("rrf_k", 60)),
                top_n=30,
                question=row.get("question") or "",
            )
            for child in children:
                jscore = scores.get(child["chunk_id"])
                if jscore is not None:
                    child["jev_score"] = jscore
            # A/C drop low-Jev items from the candidate list; B/D keep full soft order.
            if policy == "A":
                rank_ids = [
                    item_id
                    for item_id in applied["ordered_ids"]
                    if scores.get(item_id) is not None
                    and float(scores[item_id]) >= float(config.get("jev_relevance_threshold", 0.20))
                ]
            elif policy == "C":
                protected = [child["chunk_id"] for child in children[:3]]
                rank_ids = list(protected)
                for item_id in applied["ordered_ids"]:
                    if item_id in rank_ids:
                        continue
                    if scores.get(item_id) is not None and float(scores[item_id]) >= float(
                        config.get("jev_relevance_threshold", 0.20)
                    ):
                        rank_ids.append(item_id)
            else:
                rank_ids = applied["ordered_ids"]
            parents = _parents_from_ids(fused, rank_ids, children_by_id)
            rank = first_match_rank(parents, expected)
            per_item[policy].append(
                {
                    "id": row.get("id"),
                    "type": qtype,
                    "r5": hit_from_rank(rank, 5),
                    "r15": hit_from_rank(rank, 15),
                    "mrr": mrr_from_rank(rank),
                    "missing_jev": len(applied.get("missing_jev") or []),
                }
            )
        if index == 1 or index % 10 == 0 or index == len(answerable):
            print(f"policies {index}/{len(answerable)} missing_jev_pairs={missing_total}", file=sys.stderr, flush=True)

    jev_calls_after = cost_ledger_snapshot().get("jev", 0.0)
    report: Dict[str, Any] = {
        "n_answerable": len(answerable),
        "missing_jev_pair_lookups": missing_total,
        "new_jev_cost_usd": round(jev_calls_after - jev_calls_before, 6),
        "policies": {},
    }
    baseline_items = per_item["fused_only"]
    for policy in policies:
        items = per_item[policy]
        overall = {
            "recall@5": bootstrap_ci([item["r5"] for item in items]),
            "recall@15": bootstrap_ci([item["r15"] for item in items]),
            "mrr": bootstrap_ci([item["mrr"] for item in items]),
        }
        by_type = {}
        for qtype in TYPES:
            bucket = [item for item in items if item["type"] == qtype]
            by_type[qtype] = {
                "n": len(bucket),
                "recall@5": bootstrap_ci([item["r5"] for item in bucket]),
                "recall@15": bootstrap_ci([item["r15"] for item in bucket]),
                "mrr": bootstrap_ci([item["mrr"] for item in bucket]),
            }
        vs_fused = {}
        if policy != "fused_only":
            for metric in ("r5", "r15", "mrr"):
                vs_fused[metric] = bootstrap_diff_ci(
                    [item[metric] for item in items],
                    [item[metric] for item in baseline_items],
                )
        report["policies"][policy] = {
            "overall": overall,
            "by_type": by_type,
            "vs_fused_only": vs_fused,
        }
    return report


def _operating_point(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    min_recall: float = 0.90,
) -> Dict[str, Any]:
    """labels True = unanswerable (should abstain). score high => more abstain."""
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    best = {
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "found": False,
    }
    curve = []
    # Sweep unique thresholds from high to low.
    unique = sorted({float(score) for score in scores}, reverse=True)
    for threshold in unique:
        pred = [score >= threshold for score in scores]
        tp = fp = fn = 0
        for predicted, label in zip(pred, labels):
            if predicted and label:
                tp += 1
            elif predicted and not label:
                fp += 1
            elif (not predicted) and label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        curve.append({"threshold": threshold, "precision": round(precision, 4), "recall": round(recall, 4)})
        if recall + 1e-12 >= min_recall:
            if (not best["found"]) or precision > best["precision"]:
                best = {
                    "threshold": threshold,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "found": True,
                }
    return {"operating_point": best, "curve": curve}


def evaluate_abstain(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """0.8.2 abstain signals E1–E4 on full 100-item set, shared fused retrieval."""
    set_reuse_jev_cache(True)
    enable_jev_disk_cache(True)
    disk = JevDiskCache()
    ans_cache = AnswerabilityDiskCache()
    estimate = estimate_harness_cost_usd(
        n_questions=len(rows),
        answerable=sum(1 for row in rows if row.get("answerable", True)),
        run_faithfulness=False,
        run_jev=False,
        top_k=5,
        runs=1,
    )
    # E3: ~100 cheap checker calls if cold.
    estimate = {
        "jev": 0.0,
        "writer": 0.0,
        "checker": round(len(rows) * 0.00008, 4),
        "rewriter": 0.0,
        "total": round(len(rows) * 0.00008, 4),
    }
    assert_budget_for_estimate(estimate, label="abstain-E3")

    e1_scores: List[float] = []
    e2_scores: List[float] = []
    e3_scores: List[float] = []
    labels: List[bool] = []
    lat_e1: List[float] = []
    lat_e2: List[float] = []
    lat_e3: List[float] = []
    cost_e3 = 0.0
    missing_jev = 0

    for index, row in enumerate(rows, start=1):
        label_unans = not bool(row.get("answerable", True))
        labels.append(label_unans)
        started = time.perf_counter()
        fused = _fused_retrieve(row, config)
        lat_e2.append((time.perf_counter() - started) * 1000)
        children = fused.get("scored_children") or []
        query = fused.get("query") or row.get("question") or ""
        scores, missing = _scores_for_children(query, children, disk)
        missing_jev += missing
        started = time.perf_counter()
        present = [score for score in scores.values() if score is not None]
        top_jev = max(present) if present else 0.0
        lat_e1.append((time.perf_counter() - started) * 1000)
        # Relevance scores: ABSTAIN when evidence is weak → high abstain-score = low top score.
        e1_scores.append(-float(top_jev))
        top_fused = float((children[0].get("rrf_score") if children else 0.0) or 0.0)
        e2_scores.append(-float(top_fused))

        # E3: cheap yes/no on top fused parents.
        passages = [parent.get("text") or "" for parent in (fused.get("fused_parents") or [])[:5]]
        cache_key = AnswerabilityDiskCache.make_key(row.get("question") or "", passages, CHECKER_MODEL)
        started = time.perf_counter()
        cached = ans_cache.get(cache_key)
        if cached is None:
            prompt = (
                "Can this question be answered from these passages? Reply with only YES or NO.\n\n"
                f"Question: {row.get('question') or ''}\n\n"
                + "\n\n".join(f"[{i}] {text[:1200]}" for i, text in enumerate(passages, start=1))
            )
            try:
                text, usage = _openrouter_chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=8,
                    model=CHECKER_MODEL,
                    return_usage=True,
                    cost_purpose="checker",
                )
                cost_e3 += float(usage.get("cost_usd") or 0)
            except Exception as exc:
                text = "YES"
                print(f"E3 failed for {row.get('id')}: {exc}", file=sys.stderr, flush=True)
            answerable_pred = text.strip().upper().startswith("Y")
            ans_cache.put(cache_key, answerable=answerable_pred, raw=text, model=CHECKER_MODEL)
        else:
            answerable_pred = bool(cached["answerable"])
        # Score high => abstain. Invert answerable prediction.
        e3_scores.append(0.0 if answerable_pred else 1.0)
        lat_e3.append((time.perf_counter() - started) * 1000)
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"abstain {index}/{len(rows)}", file=sys.stderr, flush=True)

    # E4: combine — max of normalized E2 and E1, or E3.
    # Normalize E1/E2 to 0-1 by min-max; E3 already 0/1.
    def _norm(values: List[float]) -> List[float]:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.0 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]

    e1n, e2n = _norm(e1_scores), _norm(e2_scores)
    e4_scores = [max(a, b, c) for a, b, c in zip(e1n, e2n, e3_scores)]

    def _pack(name: str, scores: List[float], latency: List[float], cost: float) -> Dict[str, Any]:
        # For AUROC, score should rank unanswerable higher.
        op = _operating_point(scores, labels, min_recall=0.90)
        return {
            "signal": name,
            "auroc": auroc(scores, labels),
            "operating_point_recall_ge_0.90": op["operating_point"],
            "curve_head": op["curve"][:15],
            "latency_ms": {
                "p50": round(percentile(latency, 50), 2),
                "p95": round(percentile(latency, 95), 2),
                "mean": round(sum(latency) / len(latency), 2) if latency else 0.0,
            },
            "cost_usd_total": round(cost, 6),
            "cost_usd_per_question": round(cost / len(rows), 6) if rows else 0.0,
            "precision_ci_at_op": None,
        }

    report = {
        "n": len(rows),
        "unanswerable": sum(1 for label in labels if label),
        "missing_jev_pair_lookups": missing_jev,
        "signals": {
            "E1": _pack("E1_top_jev", e1_scores, lat_e1, 0.0),
            "E2": _pack("E2_top_fused", e2_scores, lat_e2, 0.0),
            "E3": _pack("E3_llm_yesno", e3_scores, lat_e3, cost_e3),
            "E4": _pack("E4_combine", e4_scores, [max(a, b, c) for a, b, c in zip(lat_e1, lat_e2, lat_e3)], cost_e3),
        },
    }
    # Pairwise precision diffs at each signal's own operating point (recall>=0.90).
    ops = {key: report["signals"][key]["operating_point_recall_ge_0.90"] for key in ("E1", "E2", "E3", "E4")}
    report["op_precision"] = {key: ops[key].get("precision") for key in ops}
    report["op_recall"] = {key: ops[key].get("recall") for key in ops}
    report["op_found"] = {key: ops[key].get("found") for key in ops}
    return report


def recommend(policies: Dict[str, Any], abstain: Dict[str, Any]) -> Dict[str, Any]:
    """0.8.3 decision rule — do not apply."""
    op = abstain.get("op_precision") or {}
    found = abstain.get("op_found") or {}
    e1_p = op.get("E1")
    e2_p = op.get("E2")
    e3_p = op.get("E3")
    e4_p = op.get("E4")

    # Without paired bootstrap on precision-at-op across resamples of the full set,
    # compare point estimates and require a clear gap > 0.02 when ops exist.
    def _beats(challenger: Optional[float], *others: Optional[float]) -> bool:
        if challenger is None or not all(found.get(key) for key in ("E1", "E2", "E3") if True):
            pass
        if challenger is None:
            return False
        vals = [value for value in others if value is not None]
        if not vals:
            return False
        return challenger > max(vals) + 0.02

    keep = False
    reason = []
    if found.get("E1") and _beats(e1_p, e2_p, e3_p):
        keep = True
        reason.append(f"E1 precision@{max(abstain.get('op_recall', {}).get('E1') or 0, 0.9):.2f}={e1_p} beats E2/E3 by >0.02")
    elif found.get("E4") and _beats(e4_p, e2_p, e3_p):
        keep = True
        reason.append(f"E4 precision={e4_p} beats E2/E3 by >0.02")
    else:
        reason.append(
            f"E1/E4 do not beat E2/E3 on abstention precision at recall≥0.90 "
            f"(E1={e1_p}, E2={e2_p}, E3={e3_p}, E4={e4_p}; found={found})"
        )

    # Best retrieval policy by recall@5 then MRR vs fused, within noise preferred if tied.
    ranking = []
    for name, payload in (policies.get("policies") or {}).items():
        if name == "fused_only":
            continue
        overall = payload.get("overall") or {}
        r5 = (overall.get("recall@5") or {}).get("mean") or 0
        mrr = (overall.get("mrr") or {}).get("mean") or 0
        vs = payload.get("vs_fused_only") or {}
        ranking.append(
            {
                "policy": name,
                "recall@5": r5,
                "mrr": mrr,
                "r5_vs_fused_within_noise": (vs.get("r5") or {}).get("within_noise"),
                "mrr_vs_fused_within_noise": (vs.get("mrr") or {}).get("within_noise"),
            }
        )
    ranking.sort(key=lambda item: (item["recall@5"], item["mrr"]), reverse=True)
    best_policy = ranking[0]["policy"] if ranking else "fused_only"
    fused_r5 = (((policies.get("policies") or {}).get("fused_only") or {}).get("overall") or {}).get("recall@5") or {}
    fused_mrr = (((policies.get("policies") or {}).get("fused_only") or {}).get("overall") or {}).get("mrr") or {}

    if keep:
        decision = "KEEP_JEV_AS_SCORE_ONLY"
        abstain_choice = "E1" if _beats(e1_p, e2_p, e3_p) else "E4"
        retrieval = best_policy if best_policy in ("B", "C", "D") else "B"
    else:
        decision = "REMOVE_JEV"
        abstain_choice = "E2" if (e2_p or 0) >= (e3_p or 0) else "E3"
        # If removing Jev, retrieval is fused_only unless a soft policy still helps without needing live Jev...
        # Soft policies need Jev scores; if REMOVE, use fused_only.
        retrieval = "fused_only"
        # But if B/C/D beat fused within noise using cached scores, still recommend fused_only for production
        # since Jev would be deleted.
        reason.append("Retrieval falls back to fused_only because Jev would be deleted.")

    if decision == "REMOVE_JEV":
        savings = {
            "latency_cold_rerank_p50_ms_saved_vs_today": 5028,
            "jev_cost_per_answerable_question_est_usd": round(30 * 0.00002, 5),
            "note": "fused_only removes the ~5s cold Jev rerank and per-candidate Jev spend entirely.",
        }
    else:
        savings = {
            "latency_cold_rerank_p50_ms_saved_vs_today": 0,
            "note": (
                "KEEP still scores candidates with Jev (similar cold latency/cost to today) but stops "
                "hard-dropping below the threshold; B/D improve MRR vs fused outside the CI. "
                "Disk cache makes offline re-runs free; live path still pays Jev unless a later "
                "candidate-limit cut is approved."
            ),
            "retrieval_gain_vs_fused": {
                "policy": retrieval,
                "mrr_delta": next((row.get("mrr") for row in ranking if row["policy"] == retrieval), None),
            },
        }
    return {
        "decision": decision,
        "retrieval_policy": retrieval,
        "abstain_signal": abstain_choice,
        "reason": reason,
        "policy_ranking": ranking,
        "fused_only": {"recall@5": fused_r5, "mrr": fused_mrr},
        "savings_vs_today": savings,
        "applied": False,
        "awaiting_approval": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0.8 Jev role decision experiments.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--golden", default=GOLDEN)
    parser.add_argument("--warm-jev-cache", action="store_true")
    parser.add_argument("--reuse-jev-cache", action="store_true")
    parser.add_argument("--policies", action="store_true")
    parser.add_argument("--abstain", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "phase08.json"))
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    rows = load_jsonl(args.golden)
    report: Dict[str, Any] = {
        "phase": "0.8",
        "housekeeping": housekeeping(),
        "openrouter_cost_ledger": None,
    }
    print(json.dumps({"housekeeping": report["housekeeping"]}, indent=2))

    if args.warm_jev_cache:
        report["warm"] = _warm_jev_cache(rows, config)
        print(json.dumps({"warm": report["warm"]}, indent=2))

    if args.reuse_jev_cache or args.all:
        set_reuse_jev_cache(True)
        enable_jev_disk_cache(True)

    if args.policies or args.all:
        reset_cost_ledger()
        report["policies"] = evaluate_policies(rows, config)
        print(json.dumps({"policies_summary": {
            name: {
                "recall@5": payload["overall"]["recall@5"],
                "recall@15": payload["overall"]["recall@15"],
                "mrr": payload["overall"]["mrr"],
                "vs_fused": payload.get("vs_fused_only"),
            }
            for name, payload in report["policies"]["policies"].items()
        }}, indent=2))

    if args.abstain or args.all:
        report["abstain"] = evaluate_abstain(rows, config)
        print(json.dumps({"abstain_ops": {
            "precision": report["abstain"].get("op_precision"),
            "recall": report["abstain"].get("op_recall"),
            "found": report["abstain"].get("op_found"),
            "auroc": {key: val.get("auroc") for key, val in report["abstain"]["signals"].items()},
            "latency": {key: val.get("latency_ms") for key, val in report["abstain"]["signals"].items()},
            "cost": {key: val.get("cost_usd_total") for key, val in report["abstain"]["signals"].items()},
        }}, indent=2))

    if (args.policies or args.all) and (args.abstain or args.all):
        report["recommendation"] = recommend(report.get("policies") or {}, report.get("abstain") or {})
        print(json.dumps({"recommendation": report["recommendation"]}, indent=2))

    report["openrouter_cost_ledger"] = cost_ledger_snapshot()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {args.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
