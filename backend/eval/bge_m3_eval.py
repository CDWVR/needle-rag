"""Offline MiniLM vs BGE-M3 (OpenRouter) throwaway-index evaluation.

Does not change production defaults or publish any index version.
Does not download model weights or start a local embedding service.

Usage:
  python backend/eval/bge_m3_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embeddings import BGE_M3_MODEL, BGE_M3_PRICE_PER_M, BgeM3EmbeddingClient
from eval.metrics import auroc, bootstrap_ci, bootstrap_diff_ci, hit_from_rank, mrr_from_rank, percentile
from eval.schema import first_match_rank, load_jsonl
from pipeline_logic import apply_retrieval_policy
from rag_engine import (
    EMBEDDING_MODEL_ID,
    JEV_MODEL,
    _JEV_COST_PER_CANDIDATE,
    _active_collection,
    _document_hash_map,
    _embed,
    _load_parent,
    assert_budget_for_estimate,
    build_throwaway_index,
    chroma_client,
    discard_index_version,
    enable_jev_disk_cache,
    openrouter_remaining_budget,
    retrieve_parents,
    set_embedding_client_override,
    set_jev_cache_enabled,
    set_reuse_jev_cache,
)
from jev_disk_cache import JevDiskCache

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "baseline_config.json")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "baselines", "bge_m3_eval.json")
TYPES = ("factual", "exact", "multihop", "followup")

# Non-sensitive test documents only (text is sent to OpenRouter providers for BGE).
SAFE_DOCUMENT_NAMES = (
    "Attention Mechanism.pdf",
    "Autoencoders.pdf",
)
SENSITIVE_DOCUMENT_NAMES = ("AI Engineering by Chip Huyen.pdf",)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sensitive_document_ids() -> List[str]:
    collection, _active = _active_collection()
    payload = collection.get(include=["metadatas"])
    ids = set()
    for metadata in payload.get("metadatas") or []:
        metadata = metadata or {}
        if metadata.get("document_name") in SENSITIVE_DOCUMENT_NAMES:
            doc_id = metadata.get("document_id") or ""
            if doc_id:
                ids.add(doc_id)
    return sorted(ids)


def _safe_embed_inputs(embed_style: str) -> List[str]:
    from pipeline_logic import contextual_passage

    collection, _active = _active_collection()
    payload = collection.get(include=["documents", "metadatas"])
    contextual = embed_style == "contextual"
    texts: List[str] = []
    for document, metadata in zip(payload.get("documents") or [], payload.get("metadatas") or []):
        metadata = metadata or {}
        if metadata.get("document_name") not in SAFE_DOCUMENT_NAMES:
            continue
        raw = metadata.get("raw_text") or document or ""
        texts.append(contextual_passage(metadata.get("header_context") or "", raw, contextual))
    return texts


def _full_corpus_embed_inputs(embed_style: str) -> List[str]:
    from pipeline_logic import contextual_passage

    collection, _active = _active_collection()
    payload = collection.get(include=["documents", "metadatas"])
    contextual = embed_style == "contextual"
    texts: List[str] = []
    for document, metadata in zip(payload.get("documents") or [], payload.get("metadatas") or []):
        metadata = metadata or {}
        raw = metadata.get("raw_text") or document or ""
        texts.append(contextual_passage(metadata.get("header_context") or "", raw, contextual))
    return texts


def _index_size_bytes(collection_name: str, dims: int, chunk_count: int) -> Dict[str, Any]:
    store_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_store")
    on_disk = 0
    # Chroma persistent segments vary by version; also report theoretical dense size.
    for root, _dirs, files in os.walk(store_dir):
        if collection_name in root:
            for name in files:
                try:
                    on_disk += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    theoretical = chunk_count * dims * 4
    return {
        "chunk_count": chunk_count,
        "dims": dims,
        "theoretical_dense_bytes": theoretical,
        "theoretical_dense_mb": round(theoretical / (1024 * 1024), 3),
        "on_disk_bytes": on_disk,
        "on_disk_mb": round(on_disk / (1024 * 1024), 3) if on_disk else None,
    }


def _parents_from_ids(
    ordered_ids: Sequence[str],
    children_by_id: Dict[str, Dict[str, Any]],
    collection_name: str,
) -> List[Dict[str, Any]]:
    hash_map = _document_hash_map(collection_name)
    parents = []
    seen = set()
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
        if key in seen:
            continue
        seen.add(key)
        parents.append(parent)
    return parents


def _fill_jev_scores(
    query: str,
    children: Sequence[Dict[str, Any]],
    disk: JevDiskCache,
    *,
    live_fill: bool,
) -> Tuple[Dict[str, Optional[float]], int]:
    out: Dict[str, Optional[float]] = {}
    missing_texts: List[Tuple[str, str]] = []
    for child in children:
        chunk_id = child["chunk_id"]
        text = child.get("text") or ""
        score = disk.get(query=query, chunk_text=text, jev_model=JEV_MODEL)
        if score is None:
            missing_texts.append((chunk_id, text))
            out[chunk_id] = None
        else:
            out[chunk_id] = score
    if missing_texts and live_fill:
        from rag_engine import _jev_scores, record_cost

        fresh = _jev_scores(query, [text for _cid, text in missing_texts])
        record_cost("jev", len(missing_texts) * _JEV_COST_PER_CANDIDATE)
        for (chunk_id, text), score in zip(missing_texts, fresh):
            disk.put(query=query, chunk_text=text, jev_model=JEV_MODEL, score=float(score))
            out[chunk_id] = float(score)
        return out, 0
    return out, len(missing_texts)


def _summarize_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    def pack(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {"n": 0}
        r5 = [float(row["r5"]) for row in subset]
        r15 = [float(row["r15"]) for row in subset]
        r30 = [float(row["r30"]) for row in subset]
        mrr = [float(row["mrr"]) for row in subset]
        return {
            "n": len(subset),
            "recall@5": round(sum(r5) / len(r5), 4),
            "recall@15": round(sum(r15) / len(r15), 4),
            "recall@30": round(sum(r30) / len(r30), 4),
            "mrr": round(sum(mrr) / len(mrr), 4),
            "recall@5_ci": bootstrap_ci(r5),
            "recall@15_ci": bootstrap_ci(r15),
            "recall@30_ci": bootstrap_ci(r30),
            "mrr_ci": bootstrap_ci(mrr),
        }

    by_type = {qtype: pack([row for row in items if row.get("type") == qtype]) for qtype in TYPES}
    return {"overall": pack(items), "by_type": by_type}


def evaluate_index(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    collection_name: str,
    exclude_document_ids: Sequence[str],
    label: str,
    live_jev_fill: bool,
) -> Dict[str, Any]:
    enable_jev_disk_cache(True)
    set_jev_cache_enabled(True)
    set_reuse_jev_cache(True)
    disk = JevDiskCache()
    answerable = [row for row in rows if row.get("answerable", True)]
    fused_items: List[Dict[str, Any]] = []
    policy_b_items: List[Dict[str, Any]] = []
    e1_scores: List[float] = []
    e1_labels: List[bool] = []
    query_embed_ms: List[float] = []
    missing_jev_total = 0
    retrieve_ms: List[float] = []

    for index, row in enumerate(rows, start=1):
        question = row.get("question") or ""
        started = time.perf_counter()
        _embed([question])
        query_embed_ms.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        fused = retrieve_parents(
            question,
            history=row.get("history") or [],
            collection_name=collection_name,
            top_k=int(config.get("top_k", 30)),
            similarity_threshold=float(config.get("similarity_threshold", 0.30)),
            rrf_k=int(config.get("rrf_k", 60)),
            max_parents=30,
            use_jev=False,
            retrieval_rerank_mode="fused_only",
            allow_retry=False,
            exclude_document_ids=list(exclude_document_ids),
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
            retry_threshold=float(config.get("retry_threshold", 0.35)),
        )
        retrieve_ms.append((time.perf_counter() - started) * 1000)
        children = fused.get("scored_children") or []
        query = fused.get("query") or question
        scores, missing = _fill_jev_scores(query, children, disk, live_fill=live_jev_fill)
        missing_jev_total += missing
        children_by_id = {child["chunk_id"]: child for child in children}
        expected = row.get("expected") or []
        qtype = row.get("type") or "factual"
        unanswerable = not bool(row.get("answerable", True))
        e1_labels.append(unanswerable)
        present = [score for score in scores.values() if score is not None]
        top_jev = max(present) if present else 0.0
        e1_scores.append(-float(top_jev))

        if not unanswerable:
            fused_rank = first_match_rank(fused.get("fused_parents") or [], expected)
            fused_items.append(
                {
                    "id": row.get("id"),
                    "type": qtype,
                    "r5": hit_from_rank(fused_rank, 5),
                    "r15": hit_from_rank(fused_rank, 15),
                    "r30": hit_from_rank(fused_rank, 30),
                    "mrr": mrr_from_rank(fused_rank),
                }
            )
            applied = apply_retrieval_policy(
                children,
                scores,
                policy="B",
                jev_threshold=float(config.get("jev_relevance_threshold", 0.20)),
                rrf_k=int(config.get("rrf_k", 60)),
                top_n=30,
                question=question,
            )
            for child in children:
                jscore = scores.get(child["chunk_id"])
                if jscore is not None:
                    child["jev_score"] = jscore
            parents = _parents_from_ids(applied["ordered_ids"], children_by_id, collection_name)
            rank = first_match_rank(parents, expected)
            policy_b_items.append(
                {
                    "id": row.get("id"),
                    "type": qtype,
                    "r5": hit_from_rank(rank, 5),
                    "r15": hit_from_rank(rank, 15),
                    "r30": hit_from_rank(rank, 30),
                    "mrr": mrr_from_rank(rank),
                }
            )

        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"[{label}] {index}/{len(rows)}", file=sys.stderr, flush=True)

    op = _operating_point(e1_scores, e1_labels, min_recall=0.90)
    return {
        "label": label,
        "n_rows": len(rows),
        "n_answerable": len(answerable),
        "missing_jev_pair_lookups": missing_jev_total,
        "fused_only": _summarize_items(fused_items),
        "policy_B": _summarize_items(policy_b_items),
        "e1": {
            "auroc": auroc(e1_scores, e1_labels),
            "precision_at_recall_ge_0.90": op["operating_point"],
        },
        "query_embed_latency_ms": {
            "p50": round(percentile(query_embed_ms, 50), 2),
            "p95": round(percentile(query_embed_ms, 95), 2),
            "mean": round(sum(query_embed_ms) / len(query_embed_ms), 2) if query_embed_ms else 0.0,
            "n": len(query_embed_ms),
        },
        "retrieve_latency_ms": {
            "p50": round(percentile(retrieve_ms, 50), 2),
            "p95": round(percentile(retrieve_ms, 95), 2),
            "mean": round(sum(retrieve_ms) / len(retrieve_ms), 2) if retrieve_ms else 0.0,
        },
        "per_item": {"fused_only": fused_items, "policy_B": policy_b_items},
    }


def _operating_point(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    min_recall: float = 0.90,
) -> Dict[str, Any]:
    best = {"threshold": None, "precision": 0.0, "recall": 0.0, "found": False}
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
        if recall + 1e-12 >= min_recall:
            if (not best["found"]) or precision > best["precision"]:
                best = {
                    "threshold": threshold,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "found": True,
                }
    return {"operating_point": best}


def recommend(minilm: Dict[str, Any], bge: Dict[str, Any], cost: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    switch = False

    def paired(metric: str, mode: str) -> Dict[str, Any]:
        left = minilm["per_item"][mode]
        right = bge["per_item"][mode]
        by_id_l = {row["id"]: row for row in left}
        by_id_r = {row["id"]: row for row in right}
        ids = [row["id"] for row in left if row["id"] in by_id_r]
        a = [float(by_id_l[i][metric]) for i in ids]
        b = [float(by_id_r[i][metric]) for i in ids]
        # bootstrap_diff_ci(left, right) => mean(left) - mean(right); pass BGE then MiniLM.
        diff = bootstrap_diff_ci(b, a) if a and b else {"diff": 0.0, "low": 0.0, "high": 0.0, "within_noise": True}
        return {
            "minilm": round(sum(a) / len(a), 4) if a else 0.0,
            "bge": round(sum(b) / len(b), 4) if b else 0.0,
            "diff_ci": diff,
            "outside_ci_improved": bool(diff.get("low", 0) > 0),
            "outside_ci_regressed": bool(diff.get("high", 0) < 0),
            "within_ci": bool(diff.get("within_noise", True)),
        }

    comparisons = {
        "fused_recall@15": paired("r15", "fused_only"),
        "fused_mrr": paired("mrr", "fused_only"),
        "policyB_recall@15": paired("r15", "policy_B"),
        "policyB_mrr": paired("mrr", "policy_B"),
    }

    improved = any(
        comparisons[key]["outside_ci_improved"]
        for key in ("fused_recall@15", "fused_mrr", "policyB_recall@15", "policyB_mrr")
    )
    regressed = any(
        comparisons[key]["outside_ci_regressed"]
        for key in ("fused_recall@15", "fused_mrr", "policyB_recall@15", "policyB_mrr")
    )
    within = all(comparisons[key]["within_ci"] for key in comparisons)

    bge_q_p95 = float(bge.get("query_embed_latency_ms", {}).get("p95") or 0)
    minilm_q_p95 = float(minilm.get("query_embed_latency_ms", {}).get("p95") or 0)
    latency_ok = bge_q_p95 <= max(250.0, minilm_q_p95 * 20)
    cost_ok = float(cost.get("eval_embed_usd") or 0) < 0.05 and float(cost.get("full_reembed_usd") or 0) < 1.0

    if improved and not regressed:
        switch = True
        reasons.append("BGE-M3 improved recall@15 or MRR outside the bootstrap 95% CI on fused and/or policy B.")
    elif within and latency_ok and cost_ok:
        switch = True
        reasons.append(
            "BGE-M3 is within CI of MiniLM on recall@15/MRR with acceptable query latency and embedding cost."
        )
    else:
        switch = False
        if regressed:
            reasons.append("BGE-M3 regressed recall@15 or MRR outside the bootstrap 95% CI.")
        if not latency_ok:
            reasons.append(f"BGE query-embed p95={bge_q_p95:.1f}ms is not acceptable vs MiniLM p95={minilm_q_p95:.1f}ms.")
        if not cost_ok:
            reasons.append("Embedding cost is not acceptable for a switch.")
        if within and not (latency_ok and cost_ok):
            reasons.append("Within-CI parity alone is not enough given latency/cost.")
        if not reasons:
            reasons.append("No clear improvement outside CI.")

    return {
        "switch_to_bge_m3": switch,
        "reasons": reasons,
        "comparisons": comparisons,
        "latency_ok": latency_ok,
        "cost_ok": cost_ok,
        "full_reembed_usd": cost.get("full_reembed_usd"),
    }


def _print_table(title: str, summary: Dict[str, Any]) -> None:
    overall = summary.get("overall") or {}
    print(f"\n### {title}")
    print(
        "| split | n | r@5 | r@5 CI | r@15 | r@15 CI | r@30 | r@30 CI | MRR | MRR CI |"
    )
    print("|---|---:|---:|---|---:|---|---:|---|---:|---|")

    def fmt_ci(ci: Any) -> str:
        if not isinstance(ci, dict):
            return "-"
        return f"[{ci.get('low')}, {ci.get('high')}]"

    def row(name: str, block: Dict[str, Any]) -> None:
        if not block or not block.get("n"):
            print(f"| {name} | 0 | - | - | - | - | - | - | - | - |")
            return
        print(
            f"| {name} | {block['n']} | {block['recall@5']:.4f} | {fmt_ci(block.get('recall@5_ci'))} | "
            f"{block['recall@15']:.4f} | {fmt_ci(block.get('recall@15_ci'))} | "
            f"{block['recall@30']:.4f} | {fmt_ci(block.get('recall@30_ci'))} | "
            f"{block['mrr']:.4f} | {fmt_ci(block.get('mrr_ci'))} |"
        )

    row("overall", overall)
    for qtype in TYPES:
        row(qtype, (summary.get("by_type") or {}).get(qtype) or {})


def design_note() -> str:
    return """
## Design note (no implementation)

### Schema
Add to `index_versions`:
- `embedding_model` TEXT NOT NULL  -- e.g. all-MiniLM-L6-v2 / baai/bge-m3
- `embedding_dim` INTEGER NOT NULL -- e.g. 384 / 1024

At query time, refuse retrieval when the active version's `(embedding_model, embedding_dim)`
does not match the process embedder (same check already sketched by `require_compatible()`,
extended to dim). This prevents mixing 384-d and 1024-d vectors in one Chroma space.

### Phase 2 cutover
1. Build a new unpublished index version with the chosen embedder (BGE via OpenRouter or TEI).
2. Run the golden regression gate (recall@15 / MRR CIs + E1 AUROC / precision@recall>=0.90).
3. Publish only if the gate passes; keep the prior version retained for rollback.
4. Rollback = re-point active `version_id` to the previous published MiniLM version (no re-embed).

### Pending
- Self-hosted TEI backend (`EMBED_BACKEND=tei`) - not implemented.
- Latency and memory sizing for TEI (batch size, GPU/CPU, p95 query embed).
- Cross-backend vector check (same text -> cosine self-sim ~1.0; dim/model assertion in CI).
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--golden", default=GOLDEN)
    parser.add_argument("--keep-indexes", action="store_true")
    parser.add_argument("--no-live-jev", action="store_true", help="Reuse disk Jev only (may leave gaps).")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = load_jsonl(args.golden)
    if len(rows) != 100:
        print(f"WARNING: golden has {len(rows)} rows; expected 100", file=sys.stderr)

    listing = BgeM3EmbeddingClient.verify_model_listing()
    print("OpenRouter BGE-M3 listing:")
    print(f"  slug: {listing.get('slug')}")
    print(f"  price_per_m_tokens_usd: {listing.get('price_per_m_tokens_usd')}")
    print(f"  configured_price_per_m: {BGE_M3_PRICE_PER_M}")
    for provider in listing.get("providers") or []:
        print(
            f"  provider: {provider.get('provider_name')} "
            f"tag={provider.get('tag')} ${provider.get('price_per_m_tokens')}/M "
            f"ctx={provider.get('context_length')}"
        )

    embed_style = config.get("embed_style") or "raw"
    chunking = config.get("chunking") or "Parent-child"
    safe_texts = _safe_embed_inputs(embed_style)
    full_texts = _full_corpus_embed_inputs(embed_style)
    bge_client = BgeM3EmbeddingClient()
    corpus_est = bge_client.estimate_cost_usd(safe_texts)
    # Query embeds for 100 questions (conservative).
    query_texts = [(row.get("question") or "") for row in rows]
    query_est = bge_client.estimate_cost_usd(query_texts)
    # Jev fill worst-case if candidates are cold: 100 * 30.
    jev_est = len(rows) * int(config.get("top_k", 30)) * _JEV_COST_PER_CANDIDATE
    full_reembed = bge_client.estimate_cost_usd(full_texts)
    estimate = {
        "jev": round(jev_est, 4),
        "writer": 0.0,
        "checker": 0.0,
        "rewriter": 0.0,
        "total": round(corpus_est + query_est + jev_est, 4),
        "bge_corpus": round(corpus_est, 6),
        "bge_queries": round(query_est, 6),
        "full_reembed_usd": round(full_reembed, 6),
    }
    print(
        f"Estimated embedding cost (safe corpus {len(safe_texts)} chunks): "
        f"${estimate['bge_corpus']:.6f}; queries=${estimate['bge_queries']:.6f}; "
        f"jev_fill<=${estimate['jev']:.4f}; full_reembed=${estimate['full_reembed_usd']:.6f}"
    )
    assert_budget_for_estimate(estimate, label="bge-m3-eval")

    exclude_ids = _sensitive_document_ids()
    print(
        f"Safe docs={list(SAFE_DOCUMENT_NAMES)}; excluding {len(exclude_ids)} sensitive document id(s) "
        f"from keyword fusion.",
        file=sys.stderr,
        flush=True,
    )

    built: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "model_listing": listing,
        "price_per_m_tokens_usd": listing.get("price_per_m_tokens_usd") or BGE_M3_PRICE_PER_M,
        "safe_documents": list(SAFE_DOCUMENT_NAMES),
        "excluded_documents": list(SENSITIVE_DOCUMENT_NAMES),
        "estimate": estimate,
        "remaining_budget_before": openrouter_remaining_budget(),
        "embed_backend_default": os.getenv("EMBED_BACKEND", "minilm"),
        "design_note": design_note(),
    }

    try:
        # --- MiniLM baseline throwaway (reuse production vectors; no publish) ---
        set_embedding_client_override(None)
        print("Building MiniLM throwaway index...", file=sys.stderr, flush=True)
        minilm_meta = build_throwaway_index(
            embed_style=embed_style,
            chunking=chunking,
            embedding_model_id=EMBEDDING_MODEL_ID,
            include_document_names=list(SAFE_DOCUMENT_NAMES),
            force_reembed=False,
        )
        built.append(minilm_meta)
        print(json.dumps({"minilm_index": minilm_meta}, indent=2), file=sys.stderr, flush=True)

        # --- BGE-M3 throwaway (OpenRouter only; override only — do not mutate EMBED_BACKEND) ---
        set_embedding_client_override(bge_client)
        print("Building BGE-M3 throwaway index via OpenRouter...", file=sys.stderr, flush=True)
        started = time.perf_counter()
        bge_meta = build_throwaway_index(
            embed_style=embed_style,
            chunking=chunking,
            embedding_model_id=BGE_M3_MODEL,
            include_document_names=list(SAFE_DOCUMENT_NAMES),
            force_reembed=True,
        )
        bge_build_s = time.perf_counter() - started
        bge_meta["providers_seen"] = list(bge_client.last_providers)
        bge_meta["embed_stats"] = dict(bge_client.stats)
        built.append(bge_meta)
        print(
            f"BGE providers this run: {Counter(bge_client.last_providers)}",
            file=sys.stderr,
            flush=True,
        )
        print(json.dumps({"bge_index": bge_meta}, indent=2), file=sys.stderr, flush=True)

        live_jev = not args.no_live_jev
        # Evaluate MiniLM
        set_embedding_client_override(None)
        minilm_eval = evaluate_index(
            rows,
            config,
            collection_name=minilm_meta["collection_name"],
            exclude_document_ids=exclude_ids,
            label="minilm",
            live_jev_fill=live_jev,
        )
        minilm_eval["corpus_embed_ms"] = minilm_meta.get("corpus_embed_ms")
        minilm_eval["corpus_embed_cost_usd"] = 0.0
        minilm_eval["index_size"] = _index_size_bytes(
            minilm_meta["collection_name"], 384, int(minilm_meta.get("chunk_count") or 0)
        )

        # Evaluate BGE
        set_embedding_client_override(bge_client)
        bge_eval = evaluate_index(
            rows,
            config,
            collection_name=bge_meta["collection_name"],
            exclude_document_ids=exclude_ids,
            label="bge-m3",
            live_jev_fill=live_jev,
        )
        bge_eval["corpus_embed_ms"] = bge_meta.get("corpus_embed_ms")
        bge_eval["corpus_embed_wall_s"] = round(bge_build_s, 2)
        bge_eval["corpus_embed_cost_usd"] = float(bge_client.stats.get("cost_usd") or 0.0)
        bge_eval["providers_seen"] = list(bge_client.last_providers)
        bge_eval["index_size"] = _index_size_bytes(
            bge_meta["collection_name"], 1024, int(bge_meta.get("chunk_count") or 0)
        )

        cost_block = {
            "eval_embed_usd": bge_eval["corpus_embed_cost_usd"],
            "full_reembed_usd": estimate["full_reembed_usd"],
            "query_embed_est_usd": estimate["bge_queries"],
        }
        decision = recommend(minilm_eval, bge_eval, cost_block)

        # Strip bulky per-item lists from printed report copy but keep in file.
        report.update(
            {
                "minilm_index": minilm_meta,
                "bge_index": bge_meta,
                "minilm": {k: v for k, v in minilm_eval.items() if k != "per_item"},
                "bge": {k: v for k, v in bge_eval.items() if k != "per_item"},
                "recommendation": decision,
                "remaining_budget_after": openrouter_remaining_budget(),
                "published": False,
                "production_defaults_changed": False,
            }
        )
        report["minilm"]["per_item"] = minilm_eval["per_item"]
        report["bge"]["per_item"] = bge_eval["per_item"]

        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

        # ---- Tables ----
        print("\n## Corpus / cost / latency")
        print("| embedder | chunks | corpus embed ms | corpus cost USD | query embed p50/p95 ms | index size (theoretical MB) | providers |")
        print("|---|---:|---:|---:|---|---:|---|")
        print(
            f"| MiniLM-L6 | {minilm_meta.get('chunk_count')} | {minilm_meta.get('corpus_embed_ms')} | 0 | "
            f"{minilm_eval['query_embed_latency_ms']['p50']}/{minilm_eval['query_embed_latency_ms']['p95']} | "
            f"{minilm_eval['index_size']['theoretical_dense_mb']} | local |"
        )
        providers = ",".join(sorted(set(bge_client.last_providers))) or "n/a"
        print(
            f"| BGE-M3 | {bge_meta.get('chunk_count')} | {bge_meta.get('corpus_embed_ms')} | "
            f"{bge_eval['corpus_embed_cost_usd']} | "
            f"{bge_eval['query_embed_latency_ms']['p50']}/{bge_eval['query_embed_latency_ms']['p95']} | "
            f"{bge_eval['index_size']['theoretical_dense_mb']} | {providers} |"
        )
        print(f"\nFull-corpus re-embed cost (all {len(full_texts)} chunks @ ${BGE_M3_PRICE_PER_M}/M): "
              f"${estimate['full_reembed_usd']:.6f}")

        _print_table("MiniLM fused-only", minilm_eval["fused_only"])
        _print_table("BGE-M3 fused-only", bge_eval["fused_only"])
        _print_table("MiniLM policy B", minilm_eval["policy_B"])
        _print_table("BGE-M3 policy B", bge_eval["policy_B"])

        print("\n### Abstain E1 (top Jev inverted)")
        print("| embedder | AUROC | precision@recall>=0.90 | recall | found |")
        print("|---|---:|---:|---:|---|")
        for name, block in (("MiniLM", minilm_eval["e1"]), ("BGE-M3", bge_eval["e1"])):
            op = block.get("precision_at_recall_ge_0.90") or {}
            print(
                f"| {name} | {block.get('auroc')} | {op.get('precision')} | {op.get('recall')} | {op.get('found')} |"
            )

        print("\n### Recommendation")
        print(f"switch_to_bge_m3: {decision['switch_to_bge_m3']}")
        for reason in decision["reasons"]:
            print(f"- {reason}")
        print(f"full_reembed_usd: {decision['full_reembed_usd']}")
        print(json.dumps(decision.get("comparisons"), indent=2))

        print("\n" + design_note())
        print(f"\nWrote {OUT_PATH}")
        return 0
    finally:
        set_embedding_client_override(None)
        if not args.keep_indexes:
            for meta in built:
                try:
                    discard_index_version(meta["version_id"], meta["collection_name"])
                    print(f"Discarded throwaway {meta['collection_name']}", file=sys.stderr, flush=True)
                except Exception as exc:
                    print(f"Failed to discard {meta}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
