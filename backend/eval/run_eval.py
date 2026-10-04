"""Golden-set harness with faithfulness, ablation, latency, and threshold sweeps.

Usage from the repo root:
  python backend/eval/run_eval.py --config backend/eval/baseline_config.json
  python backend/eval/run_eval.py --faithfulness --ablation
  python backend/eval/run_eval.py --sweep
  python backend/eval/run_eval.py --runs 3 --out backend/eval/baselines/phase0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.metrics import (
    abstention_scores,
    hit_from_rank,
    mean_spread,
    mrr_from_rank,
    summarize_latencies,
)
from eval.schema import first_match_rank, load_jsonl
from pipeline_logic import infra_error_share, infra_errors_exceed_share
from rag_engine import (
    CHECKER_MODEL,
    EVAL_MAX_INFRA_SHARE,
    JEV_MODEL,
    OPENROUTER_MODEL,
    RETRY_THRESHOLD,
    answer_for_eval,
    assert_budget_for_estimate,
    build_throwaway_index,
    clear_jev_cache,
    cost_ledger_snapshot,
    diagnose_answer,
    discard_index_version,
    enable_jev_disk_cache,
    env_defaults,
    estimate_harness_cost_usd,
    index_status,
    jev_cache_stats,
    reset_cost_ledger,
    retrieve_parents,
    set_jev_cache_enabled,
    set_reuse_jev_cache,
)


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "baseline_config.json")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")
STAGES = ("condense", "retrieve", "rerank", "write", "check")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settings_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    status = index_status()
    defaults = env_defaults()
    return {
        "config": config,
        "env_defaults": defaults,
        "active_index": {
            "version_id": status.get("version_id"),
            "collection_name": status.get("collection_name"),
            "embed_style": status.get("embed_style"),
            "chunking": status.get("chunking"),
            "embedding_model": status.get("embedding_model"),
        },
        "models": {
            "answer_model": OPENROUTER_MODEL,
            "checker_model": CHECKER_MODEL,
            "jev_model": JEV_MODEL,
        },
        "thresholds": {
            "top_k": config.get("top_k", defaults["top_k"]),
            "similarity_threshold": config.get("similarity_threshold", defaults["similarity_threshold"]),
            "rrf_k": config.get("rrf_k", defaults["rrf_k"]),
            "max_parents": config.get("max_parents", defaults["max_parents"]),
            "jev_relevance_threshold": config.get(
                "jev_relevance_threshold", defaults["jev_relevance_threshold"]
            ),
            "retry_threshold": config.get("retry_threshold", defaults["retry_threshold"]),
        },
    }


def _retrieval_metrics(rows: List[Dict[str, Any]], *, use_jev: bool, config: Dict[str, Any], collection_name: Optional[str] = None) -> Dict[str, Any]:
    k = int(config.get("k", 5))
    answerable = [row for row in rows if row.get("answerable", True)]
    recalls = []
    recalls_30 = []
    mrrs = []
    latency_rows = []
    for index, row in enumerate(answerable, start=1):
        result = retrieve_parents(
            row.get("question") or "",
            history=row.get("history") or [],
            collection_name=collection_name,
            top_k=int(config.get("top_k", 30)),
            similarity_threshold=float(config.get("similarity_threshold", 0.30)),
            rrf_k=int(config.get("rrf_k", 60)),
            max_parents=max(int(config.get("max_parents", 5)), 30),
            use_jev=use_jev,
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
            retry_threshold=float(config.get("retry_threshold", RETRY_THRESHOLD)),
        )
        rank = first_match_rank(result.get("ordered_parents") or result.get("parents") or [], row.get("expected") or [])
        recalls.append(hit_from_rank(rank, k))
        recalls_30.append(hit_from_rank(rank, 30))
        mrrs.append(mrr_from_rank(rank))
        latency_rows.append(result.get("latencies_ms") or {})
        if index == 1 or index % 10 == 0 or index == len(answerable):
            print(
                f"retrieval[{('jev' if use_jev else 'fused')}] {index}/{len(answerable)}",
                file=sys.stderr,
                flush=True,
            )
    return {
        "mode": "jev" if use_jev else "fused",
        "questions": len(answerable),
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "recall_at_5": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "recall_at_30": round(sum(recalls_30) / len(recalls_30), 4) if recalls_30 else None,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else None,
        "latency": summarize_latencies(latency_rows, ("condense", "retrieve", "rerank")),
    }


def _faithfulness_and_abstention(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    run_checker: bool,
    max_infra_share: float = EVAL_MAX_INFRA_SHARE,
) -> Dict[str, Any]:
    grounded_flags = []
    abstain_pred = []
    abstain_label = []
    latency_rows = []
    costs = []
    token_calls = []
    infra_count = 0
    infra_kinds: Dict[str, int] = {}
    scored_n = 0
    for index, row in enumerate(rows, start=1):
        result = answer_for_eval(
            row.get("question") or "",
            history=row.get("history") or [],
            top_k=int(config.get("top_k", 30)),
            similarity_threshold=float(config.get("similarity_threshold", 0.30)),
            rrf_k=int(config.get("rrf_k", 60)),
            max_parents=int(config.get("max_parents", 5)),
            use_jev=True,
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
            retry_threshold=float(config.get("retry_threshold", RETRY_THRESHOLD)),
            run_checker=run_checker,
        )
        if result.get("infra_error") or result.get("reject_category") == "infra_error":
            infra_count += 1
            kind = str(result.get("infra_kind") or "infra_error")
            infra_kinds[kind] = infra_kinds.get(kind, 0) + 1
            latency_rows.append(result.get("latencies_ms") or {})
            if index == 1 or index % 5 == 0 or index == len(rows):
                print(f"answer {index}/{len(rows)} infra_error={kind}", file=sys.stderr, flush=True)
            continue
        scored_n += 1
        # Faithfulness is the checker grounded rate on drafts that reached the checker.
        if row.get("answerable", True) and result.get("checked"):
            grounded_flags.append(1.0 if result.get("grounded") else 0.0)
        abstain_pred.append(bool(result.get("abstained")))
        abstain_label.append(not bool(row.get("answerable", True)))
        latency_rows.append(result.get("latencies_ms") or {})
        costs.append(float(result.get("cost_usd") or 0))
        token_calls.extend(result.get("tokens") or [])
        if index == 1 or index % 5 == 0 or index == len(rows):
            print(
                f"answer {index}/{len(rows)} abstain={result.get('abstained')} grounded={result.get('grounded')}",
                file=sys.stderr,
                flush=True,
            )
    total = len(rows)
    share = infra_error_share(infra_count, total)
    if infra_errors_exceed_share(infra_count, total, max_share=max_infra_share):
        raise RuntimeError(
            f"Harness failed: infra_error share {share:.2%} exceeds max {max_infra_share:.2%} "
            f"({infra_count}/{total})."
        )
    abstention = abstention_scores(abstain_pred, abstain_label)
    return {
        "faithfulness_grounded_rate": round(sum(grounded_flags) / len(grounded_flags), 4) if grounded_flags else None,
        "faithfulness_n": len(grounded_flags),
        "abstention": abstention,
        "unanswerable_n": sum(1 for row in rows if not row.get("answerable", True)),
        "infra_errors": infra_count,
        "infra_error_share": round(share, 4),
        "infra_kinds": infra_kinds,
        "scored_n": scored_n,
        "latency": summarize_latencies(latency_rows, STAGES),
        "mean_cost_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
        "total_cost_usd": round(sum(costs), 6),
        "token_calls": len(token_calls),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in token_calls),
        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in token_calls),
    }


def _threshold_sweep(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    answerable = [row for row in rows if row.get("answerable", True)]
    # Stratified sample keeps the sweep affordable while still covering types.
    sample = []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in answerable:
        by_type.setdefault(row.get("type") or "factual", []).append(row)
    for bucket in by_type.values():
        sample.extend(bucket[: max(1, 20 // max(1, len(by_type)))])
    sample = sample[:20] or answerable[:20]
    print(f"threshold sweep on {len(sample)} answerable items", file=sys.stderr, flush=True)
    jev_values = [round(x, 2) for x in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]]
    retry_values = [round(x, 2) for x in [0.25, 0.30, 0.35, 0.40, 0.45]]
    jev_rows = []
    for threshold in jev_values:
        local = dict(config)
        local["jev_relevance_threshold"] = threshold
        metrics = _retrieval_metrics(sample, use_jev=True, config=local)
        jev_rows.append({"jev_relevance_threshold": threshold, "recall_at_k": metrics["recall_at_k"], "mrr": metrics["mrr"]})
    retry_rows = []
    for threshold in retry_values:
        local = dict(config)
        local["retry_threshold"] = threshold
        metrics = _retrieval_metrics(sample, use_jev=True, config=local)
        retry_rows.append({"retry_threshold": threshold, "recall_at_k": metrics["recall_at_k"], "mrr": metrics["mrr"]})
    return {"sample_size": len(sample), "jev_relevance_threshold": jev_rows, "retry_threshold": retry_rows}


def run_once(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    args,
    *,
    include_sweep: bool = False,
    include_ablation: bool = True,
    include_faithfulness: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    with_jev = _retrieval_metrics(rows, use_jev=True, config=config)
    report: Dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings_snapshot(config),
        "golden_questions": len(rows),
        "with_jev": with_jev,
        "recall_at_k": with_jev["recall_at_k"],
        "recall_at_5": with_jev.get("recall_at_5"),
        "recall_at_30": with_jev.get("recall_at_30"),
        "mrr": with_jev["mrr"],
    }
    if args.ablation and include_ablation:
        without = _retrieval_metrics(rows, use_jev=False, config=config)
        report["ablation"] = {"with_jev": with_jev, "fused_only": without}
    if args.faithfulness and include_faithfulness:
        max_infra = float(getattr(args, "max_infra_share", EVAL_MAX_INFRA_SHARE))
        report["answer_quality"] = _faithfulness_and_abstention(
            rows, config, run_checker=True, max_infra_share=max_infra
        )
        report["faithfulness"] = report["answer_quality"]["faithfulness_grounded_rate"]
        report["abstention"] = report["answer_quality"]["abstention"]
        report["infra_errors"] = report["answer_quality"]["infra_errors"]
        report["infra_error_share"] = report["answer_quality"]["infra_error_share"]
        report["p50_latency_ms"] = report["answer_quality"]["latency"].get("write", {}).get("p50_ms")
        report["p95_latency_ms"] = max(
            (stage.get("p95_ms") or 0) for stage in report["answer_quality"]["latency"].values()
        )
        report["mean_cost_usd"] = report["answer_quality"]["mean_cost_usd"]
    else:
        report["faithfulness"] = None
    if args.sweep and include_sweep:
        report["threshold_sweep"] = _threshold_sweep(rows, config)
    report["elapsed_s"] = round(time.perf_counter() - started, 2)
    return report


def run_contextual_delta(rows: List[Dict[str, Any]], config: Dict[str, Any], *, runs: int = 1) -> Dict[str, Any]:
    raw_scores = []
    contextual_scores = []
    throwaway = None
    try:
        for index in range(max(1, runs)):
            clear_jev_cache()
            try:
                baseline = _retrieval_metrics(rows, use_jev=True, config=config)
            except Exception as exc:
                print(f"raw retrieval failed on run {index + 1}: {exc}", file=sys.stderr, flush=True)
                continue
            raw_scores.append(baseline)
            if throwaway is None:
                throwaway = build_throwaway_index(
                    embed_style="contextual",
                    chunking=config.get("chunking") or "Parent-child",
                )
            clear_jev_cache()
            try:
                contextual = _retrieval_metrics(
                    rows,
                    use_jev=True,
                    config=config,
                    collection_name=throwaway["collection_name"],
                )
            except Exception as exc:
                print(f"contextual retrieval failed on run {index + 1}: {exc}", file=sys.stderr, flush=True)
                continue
            contextual_scores.append(contextual)
            print(f"contextual delta run {index + 1}/{runs}", file=sys.stderr, flush=True)
        raw_recall = mean_spread([row["recall_at_k"] or 0 for row in raw_scores])
        raw_mrr = mean_spread([row["mrr"] or 0 for row in raw_scores])
        ctx_recall = mean_spread([row["recall_at_k"] or 0 for row in contextual_scores])
        ctx_mrr = mean_spread([row["mrr"] or 0 for row in contextual_scores])
        delta_recall = [
            (ctx["recall_at_k"] or 0) - (raw["recall_at_k"] or 0)
            for raw, ctx in zip(raw_scores, contextual_scores)
        ]
        delta_mrr = [
            (ctx["mrr"] or 0) - (raw["mrr"] or 0)
            for raw, ctx in zip(raw_scores, contextual_scores)
        ]
        return {
            "published": False,
            "throwaway_version": throwaway["version_id"] if throwaway else None,
            "runs": runs,
            "raw": {"recall_at_k": raw_recall, "mrr": raw_mrr, "last": raw_scores[-1] if raw_scores else None},
            "contextual": {
                "recall_at_k": ctx_recall,
                "mrr": ctx_mrr,
                "last": contextual_scores[-1] if contextual_scores else None,
            },
            "delta_recall_at_k": mean_spread(delta_recall),
            "delta_mrr": mean_spread(delta_mrr),
        }
    finally:
        if throwaway:
            discard_index_version(throwaway["version_id"], throwaway["collection_name"])


def run_diagnose(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    answerable = [row for row in rows if row.get("answerable", True)]
    histogram: Dict[str, int] = {}
    examples = []
    for index, row in enumerate(answerable, start=1):
        item = diagnose_answer(
            row.get("question") or "",
            history=row.get("history") or [],
            top_k=int(config.get("top_k", 30)),
            similarity_threshold=float(config.get("similarity_threshold", 0.30)),
            rrf_k=int(config.get("rrf_k", 60)),
            max_parents=int(config.get("max_parents", 5)),
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
            retry_threshold=float(config.get("retry_threshold", RETRY_THRESHOLD)),
        )
        item["id"] = row.get("id")
        item["type"] = row.get("type")
        histogram[item["category"]] = histogram.get(item["category"], 0) + 1
        examples.append(item)
        print(f"diagnose {index}/{len(answerable)} -> {item['category']}", file=sys.stderr, flush=True)
    # Worst examples: prefer deterministic / unparseable / dropped / ungrounded over accepted.
    severity = {
        "deterministic:citation_index": 100,
        "deterministic:quote_span": 95,
        "deterministic:number": 90,
        "injection_scan": 85,
        "checker_unparseable": 80,
        "all_sentences_dropped": 75,
        "checker_ungrounded": 70,
        "checker_unsafe": 65,
        "checker_irrelevant": 60,
        "retrieval_abstain": 40,
        "accepted": 0,
    }
    worst = sorted(examples, key=lambda item: severity.get(item["category"], 50), reverse=True)[:5]
    report = {
        "answerable": len(answerable),
        "histogram": dict(sorted(histogram.items(), key=lambda pair: (-pair[1], pair[0]))),
        "worst_examples": [
            {
                "id": item.get("id"),
                "category": item["category"],
                "question": item["question"],
                "writer_passages": item["writer_passages"],
                "draft": item["draft"],
                "checker_passages": item["checker_passages"],
                "raw_checker": item["raw_checker"],
                "reason": item["reason"],
                "final_decision": "accepted" if item["passed"] else f"rejected:{item['category']}",
                "deterministic_details": item.get("deterministic_details"),
                "flagged": item.get("flagged"),
                "passages_identical": item.get("passages_identical"),
            }
            for item in worst
        ],
    }
    print("\nHistogram:")
    for key, value in report["histogram"].items():
        print(f"  {key}: {value}")
    print("\nWorst examples:")
    print(json.dumps(report["worst_examples"], indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Needle golden-set harness.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--golden", default=GOLDEN)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--faithfulness", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--contextual-delta", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--cold-jev-cache", action="store_true", help="Disable in-memory Jev score cache for baseline runs.")
    parser.add_argument("--reuse-jev-cache", action="store_true", help="Use on-disk Jev scores only; never fetch missing pairs.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-infra-share", type=float, default=EVAL_MAX_INFRA_SHARE)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    if args.k is not None:
        config["k"] = args.k
    rows = load_jsonl(args.golden)
    if not rows:
        print("No golden questions yet. Review candidates into backend/eval/golden.jsonl.")
        return

    if args.diagnose:
        report = run_diagnose(rows, config)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        return

    answerable_n = sum(1 for row in rows if row.get("answerable", True))
    estimate = estimate_harness_cost_usd(
        n_questions=len(rows),
        answerable=answerable_n,
        run_faithfulness=bool(args.faithfulness),
        run_jev=not args.reuse_jev_cache,
        top_k=int(config.get("top_k", 30)),
        runs=max(1, args.runs),
    )
    assert_budget_for_estimate(estimate, label="run_eval")
    reset_cost_ledger()
    enable_jev_disk_cache(True)

    if args.reuse_jev_cache:
        set_reuse_jev_cache(True)
        set_jev_cache_enabled(True)
        print("Reusing on-disk Jev cache (no new Jev fetches for misses)", file=sys.stderr, flush=True)
    elif args.cold_jev_cache:
        set_jev_cache_enabled(False)
        clear_jev_cache()
        print("Jev score cache disabled for this harness run", file=sys.stderr, flush=True)
    else:
        set_jev_cache_enabled(True)

    runs = []
    total_runs = max(1, args.runs)
    for index in range(total_runs):
        if args.cold_jev_cache:
            clear_jev_cache()
        # Noise-floor repeats keep retrieval+faithfulness; sweep/ablation once is enough.
        runs.append(
            run_once(
                rows,
                config,
                args,
                include_sweep=args.sweep and index == 0,
                include_ablation=args.ablation and index == 0,
                include_faithfulness=args.faithfulness,
            )
        )
        print(f"Completed run {index + 1}/{total_runs}", file=sys.stderr)
    settings = settings_snapshot(config)
    payload: Dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "harness": "backend/eval/run_eval.py",
        "commit": _git_commit(),
        "golden_file": os.path.relpath(args.golden).replace("\\", "/"),
        "golden_sha256": _sha256_file(args.golden),
        "config_file": os.path.relpath(args.config).replace("\\", "/") if args.config else None,
        "config_snapshot": config,
        "model_slugs": settings.get("models"),
        "runs": runs,
        "aggregate": {
            "recall_at_k": mean_spread([run.get("recall_at_k") or 0 for run in runs]),
            "recall_at_5": mean_spread([run.get("recall_at_5") or run.get("recall_at_k") or 0 for run in runs]),
            "recall_at_30": mean_spread([run.get("recall_at_30") or 0 for run in runs]),
            "mrr": mean_spread([run.get("mrr") or 0 for run in runs]),
            "faithfulness": mean_spread([run.get("faithfulness") or 0 for run in runs if run.get("faithfulness") is not None])
            if any(run.get("faithfulness") is not None for run in runs)
            else None,
            "abstention_recall": mean_spread(
                [
                    ((run.get("abstention") or {}).get("recall") or 0)
                    for run in runs
                    if run.get("abstention") is not None
                ]
            )
            if any(run.get("abstention") is not None for run in runs)
            else None,
            "abstention_precision": mean_spread(
                [
                    ((run.get("abstention") or {}).get("precision") or 0)
                    for run in runs
                    if run.get("abstention") is not None
                ]
            )
            if any(run.get("abstention") is not None for run in runs)
            else None,
            "infra_errors": mean_spread([run.get("infra_errors") or 0 for run in runs]),
            "infra_error_share": mean_spread([run.get("infra_error_share") or 0 for run in runs]),
            "mean_cost_usd": mean_spread([run.get("mean_cost_usd") or 0 for run in runs if "mean_cost_usd" in run])
            if any("mean_cost_usd" in run for run in runs)
            else None,
        },
        "settings": settings,
        "golden_questions": len(rows),
        "parity_gate": {
            "module": "backend/eval/parity.py",
            "metric_tolerance": 0.02,
            "min_top5_overlap": 0.90,
            "metrics": ["recall_at_5", "recall_at_30", "mrr"],
        },
    }
    # Convenience top-level fields are aggregate mean ± spread (never a lone "best").
    payload["recall_at_k"] = payload["aggregate"]["recall_at_k"]
    payload["recall_at_5"] = payload["aggregate"]["recall_at_5"]
    payload["recall_at_30"] = payload["aggregate"]["recall_at_30"]
    payload["mrr"] = payload["aggregate"]["mrr"]
    payload["faithfulness"] = payload["aggregate"]["faithfulness"]
    if payload["aggregate"]["abstention_recall"] is not None:
        payload["abstention"] = {
            "recall": payload["aggregate"]["abstention_recall"],
            "precision": payload["aggregate"]["abstention_precision"],
        }
    elif runs and runs[0].get("answer_quality"):
        payload["abstention"] = runs[0]["answer_quality"]["abstention"]
    payload["infra_errors"] = payload["aggregate"]["infra_errors"]
    payload["infra_error_share"] = payload["aggregate"]["infra_error_share"]
    if runs and runs[0].get("answer_quality"):
        payload["latency"] = runs[0]["answer_quality"]["latency"]
        payload["p95_latency_ms"] = runs[0].get("p95_latency_ms")
        payload["mean_cost_usd"] = payload["aggregate"]["mean_cost_usd"]
    if runs and runs[0].get("ablation"):
        payload["ablation"] = runs[0]["ablation"]
    if runs and runs[0].get("threshold_sweep"):
        payload["threshold_sweep"] = runs[0]["threshold_sweep"]

    payload["jev_cache"] = jev_cache_stats()
    payload["openrouter_cost_by_purpose"] = cost_ledger_snapshot()
    if args.cold_jev_cache and runs:
        rerank = (((runs[0].get("with_jev") or {}).get("latency") or {}).get("rerank") or {})
        payload["cold_jev_rerank_latency"] = rerank
    if args.contextual_delta:
        try:
            payload["contextual_delta"] = run_contextual_delta(rows, config, runs=max(1, args.runs))
        except Exception as exc:
            payload["contextual_delta"] = {"error": str(exc), "published": False}
            print(f"contextual delta failed: {exc}", file=sys.stderr, flush=True)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
