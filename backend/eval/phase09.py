"""Phase 0.9: full-corpus MiniLM vs BGE-M3 comparison (no publish, no default changes).

Pre-declared decision rule is written into the report BEFORE any retrieval metrics
are computed.

Usage:
  python backend/eval/phase09.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embeddings import (
    BGE_M3_PRICE_PER_M,
    BgeM3EmbeddingClient,
    RoutingBgeM3Client,
    TeiEmbeddingClient,
    assert_cross_backend_cosine,
)
from eval.metrics import auroc, bootstrap_ci, bootstrap_diff_ci, hit_from_rank, mrr_from_rank, percentile
from eval.schema import first_match_rank, load_jsonl
from pipeline_logic import apply_retrieval_policy, contextual_passage
from rag_engine import (
    EMBEDDING_MODEL_ID,
    JEV_MODEL,
    _JEV_COST_PER_CANDIDATE,
    _active_collection,
    _document_hash_map,
    _embed,
    _jev_scores,
    _load_parent,
    assert_budget_for_estimate,
    build_throwaway_index,
    discard_index_version,
    enable_jev_disk_cache,
    openrouter_remaining_budget,
    record_cost,
    retrieve_parents,
    set_embedding_client_override,
    set_jev_cache_enabled,
    set_reuse_jev_cache,
)
from jev_disk_cache import JevDiskCache

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "baseline_config.json")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "baselines", "phase09.json")
TYPES = ("factual", "exact", "multihop", "followup")
BOOTSTRAP_N = 2000

# Document sensitivity flags (text for OpenRouter-safe docs may leave the machine).
SENSITIVE_DOCUMENT_NAMES = frozenset({"AI Engineering by Chip Huyen.pdf"})
SAFE_DOCUMENT_NAMES = frozenset(
    {
        "Attention Mechanism.pdf",
        "Autoencoders.pdf",
        "rnn_basics.txt",
        "information_retrieval_basics.txt",
    }
)

DECISION_RULE = {
    "primary_metrics": [
        "policy_B.recall@5",
        "policy_B.mrr",
        "policy_B.exact.recall@5",
        "policy_B.exact.mrr",
    ],
    "secondary_metrics": [
        "policy_B.recall@15",
        "policy_B.recall@30",
        "fused_only.*",
        "e1.auroc",
        "query_embed_latency",
    ],
    "method": (
        "Paired bootstrap on per-question differences (BGE-M3 minus MiniLM), "
        f"{BOOTSTRAP_N} resamples, 95% CI. Do not compare overlapping CIs of absolute means."
    ),
    "switch_if": (
        "Primary metrics are non-negative (CI lower bound > -0.01) with at least one "
        "improved outside zero (CI lower bound > 0), AND exact-type recall@5 does not "
        "regress (its CI lower bound > -0.01). Otherwise keep MiniLM or run a bigger test."
    ),
    "multiple_comparison_note": (
        "Several primary cells are inspected; family-wise error is uncontrolled. "
        "Treat borderline CI crossings cautiously."
    ),
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def corpus_inventory() -> Dict[str, Any]:
    collection, active = _active_collection()
    payload = collection.get(include=["metadatas"])
    by_doc: Dict[str, int] = Counter()
    sensitivity: Dict[str, str] = {}
    for metadata in payload.get("metadatas") or []:
        metadata = metadata or {}
        name = metadata.get("document_name") or "(unknown)"
        by_doc[name] += 1
        if name in SENSITIVE_DOCUMENT_NAMES:
            sensitivity[name] = "sensitive"
        elif name in SAFE_DOCUMENT_NAMES:
            sensitivity[name] = "safe"
        else:
            sensitivity[name] = "unlisted_treat_as_sensitive"
    return {
        "active_collection": active.get("collection_name"),
        "chunk_count": sum(by_doc.values()),
        "document_count": len(by_doc),
        "chunks_by_document": dict(by_doc),
        "sensitivity": sensitivity,
        "flag": (
            "SENSITIVE: AI Engineering by Chip Huyen.pdf must not be sent to OpenRouter. "
            "SAFE academic/fixture texts may use OpenRouter. Unlisted docs default to TEI."
        ),
    }


def golden_coverage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    collection, _active = _active_collection()
    payload = collection.get(include=["metadatas"])
    hash_by_doc: Dict[str, set] = defaultdict(set)
    doc_by_hash: Dict[str, str] = {}
    hash_map = _document_hash_map(_active_collection()[1]["collection_name"])
    for metadata in payload.get("metadatas") or []:
        metadata = metadata or {}
        doc_id = metadata.get("document_id") or ""
        name = metadata.get("document_name") or ""
        content_hash = hash_map.get(doc_id) or metadata.get("content_hash") or ""
        if content_hash:
            hash_by_doc[name].add(content_hash)
            doc_by_hash[content_hash] = name

    answerable = [row for row in rows if row.get("answerable", True)]
    docs_hit = Counter()
    for row in answerable:
        for item in row.get("expected") or []:
            name = doc_by_hash.get(item.get("content_hash") or "")
            if name:
                docs_hit[name] += 1
    return {
        "n_total": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(rows) - len(answerable),
        "answerable_by_document": dict(docs_hit),
        "documents_with_answerable": len(docs_hit),
        "corpus_documents": sorted(hash_by_doc),
        "meets_150_answerable": len(answerable) >= 150,
        "meets_5_documents": len(docs_hit) >= 5,
    }


def classify_unanswerable(row: Dict[str, Any]) -> str:
    """Split near-miss (topic-related) vs off-topic when subtype missing (0.8.4a)."""
    kind = (row.get("unanswerable_kind") or row.get("unanswerable_subtype") or "").strip().lower()
    if kind in {"near_miss", "near-miss", "nearmiss"}:
        return "near_miss"
    if kind in {"off_topic", "off-topic", "offtopic"}:
        return "off_topic"
    question = (row.get("question") or "").lower()
    off_markers = (
        "capital of",
        "new zealand",
        "population of mars",
        "olympic medal",
        "premier league",
        "stock price of",
    )
    if any(marker in question for marker in off_markers):
        return "off_topic"
    return "near_miss"


def _chunk_rows(embed_style: str) -> List[Dict[str, Any]]:
    collection, _active = _active_collection()
    payload = collection.get(include=["documents", "metadatas"])
    contextual = embed_style == "contextual"
    rows = []
    for document, metadata in zip(payload.get("documents") or [], payload.get("metadatas") or []):
        metadata = dict(metadata or {})
        raw = metadata.get("raw_text") or document or ""
        name = metadata.get("document_name") or ""
        backend = "openrouter" if name in SAFE_DOCUMENT_NAMES else "tei"
        rows.append(
            {
                "chunk_id": metadata.get("chunk_id") or "",
                "document_name": name,
                "document_id": metadata.get("document_id") or "",
                "backend": backend,
                "text": contextual_passage(metadata.get("header_context") or "", raw, contextual),
            }
        )
    return rows


class RoutedOverride:
    """Adapter used as set_embedding_client_override for corpus build + queries."""

    model_id = "baai/bge-m3"
    dimensions = 1024

    def __init__(self, router: RoutingBgeM3Client, *, query_backend: str = "tei"):
        self.router = router
        self.query_backend = query_backend
        self.query_latencies_ms: List[float] = []

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # Query path has no document names; use TEI (self-hosted) for all queries.
        started = time.perf_counter()
        if len(texts) == 1:
            out = self.router.embed_routed(texts, [self.query_backend])
            self.query_latencies_ms.append((time.perf_counter() - started) * 1000)
            return out
        # Batch corpus embeds should be pre-routed; default TEI for safety.
        return self.router.embed_routed(texts, [self.query_backend] * len(texts))


def build_bge_full_index(
    *,
    embed_style: str,
    chunking: str,
    router: RoutingBgeM3Client,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build unpublished full-corpus BGE index with per-document backend routing."""
    from rag_engine import chroma_client, store
    import uuid

    chunk_rows = _chunk_rows(embed_style)
    inventory = Counter(row["backend"] for row in chunk_rows)
    # Embed in backend buckets for correct routing + reporting.
    vectors: List[Optional[List[float]]] = [None] * len(chunk_rows)
    started = time.perf_counter()
    for backend in ("tei", "openrouter"):
        indexes = [i for i, row in enumerate(chunk_rows) if row["backend"] == backend]
        if not indexes:
            continue
        texts = [chunk_rows[i]["text"] for i in indexes]
        batch_size = router.tei.batch_size if backend == "tei" else (
            router.openrouter.batch_size if router.openrouter else 32
        )
        print(
            f"Embedding {len(indexes)} chunks via {backend} (batch_size={batch_size})",
            file=sys.stderr,
            flush=True,
        )
        embedded = router.embed_routed(texts, [backend] * len(texts))
        for offset, index in enumerate(indexes):
            vectors[index] = embedded[offset]
    embed_ms = (time.perf_counter() - started) * 1000

    # Build throwaway collection manually to attach custom vectors.
    active = store.ensure_active_version(EMBEDDING_MODEL_ID)
    old = chroma_client.get_collection(active["collection_name"])
    payload = old.get(include=["documents", "metadatas"])
    ids = list(payload.get("ids") or [])
    documents = list(payload.get("documents") or [])
    metadatas = list(payload.get("metadatas") or [])
    if len(ids) != len(vectors):
        raise RuntimeError(f"chunk/vector mismatch: {len(ids)} vs {len(vectors)}")

    version_id = str(uuid.uuid4())
    collection_name = "needle_" + version_id.replace("-", "")[:12]
    store.begin_version(version_id, "baai/bge-m3", collection_name, embed_style=embed_style, chunking=chunking)
    created = False
    try:
        new_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        created = True
        stored_meta = []
        stored_docs = []
        for document, metadata in zip(documents, metadatas):
            metadata = dict(metadata or {})
            raw = metadata.get("raw_text") or document or ""
            metadata["raw_text"] = raw
            metadata["embed_style"] = embed_style
            metadata["chunking"] = chunking
            metadata["version_id"] = version_id
            metadata["embedding_model"] = "baai/bge-m3"
            stored_docs.append(raw)
            stored_meta.append(metadata)
        for start in range(0, len(ids), 100):
            new_collection.add(
                ids=ids[start : start + 100],
                documents=stored_docs[start : start + 100],
                metadatas=stored_meta[start : start + 100],
                embeddings=vectors[start : start + 100],
            )
        if new_collection.count() != len(ids):
            raise RuntimeError("BGE throwaway count mismatch")
        meta = {
            "version_id": version_id,
            "collection_name": collection_name,
            "embed_style": embed_style,
            "chunking": chunking,
            "embedding_model": "baai/bge-m3",
            "published": False,
            "chunk_count": len(ids),
            "document_count": len({(m or {}).get("document_name") for m in metadatas}),
            "corpus_embed_ms": round(embed_ms, 2),
            "backend_chunk_counts": dict(inventory),
            "routing": dict(router.last_routing),
            "tei_batch_size": router.tei.batch_size,
            "openrouter_batch_size": (router.openrouter.batch_size if router.openrouter else None),
            "openrouter_cost_usd": float((router.openrouter.stats if router.openrouter else {}).get("cost_usd") or 0),
        }
        return meta, {"chunk_rows": len(chunk_rows)}
    except Exception:
        store.fail_version(version_id)
        if created:
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                pass
        raise


def _parents_from_ids(ordered_ids, children_by_id, collection_name):
    hash_map = _document_hash_map(collection_name)
    parents = []
    seen = set()
    for chunk_id in ordered_ids:
        child = children_by_id.get(chunk_id)
        if not child:
            continue
        parent = _load_parent(child)
        parent["content_hash"] = hash_map.get(parent.get("document_id") or "", "")
        key = parent.get("parent_id") or parent.get("text")
        if key in seen:
            continue
        seen.add(key)
        parents.append(parent)
    return parents


def _fill_jev(query, children, disk, live_fill=True):
    out = {}
    missing = []
    for child in children:
        cid = child["chunk_id"]
        text = child.get("text") or ""
        score = disk.get(query=query, chunk_text=text, jev_model=JEV_MODEL)
        if score is None:
            missing.append((cid, text))
            out[cid] = None
        else:
            out[cid] = score
    if missing and live_fill:
        fresh = _jev_scores(query, [text for _c, text in missing])
        record_cost("jev", len(missing) * _JEV_COST_PER_CANDIDATE)
        for (cid, text), score in zip(missing, fresh):
            disk.put(query=query, chunk_text=text, jev_model=JEV_MODEL, score=float(score))
            out[cid] = float(score)
        return out, 0
    return out, len(missing)


def evaluate_index(rows, config, *, collection_name, label, query_embed_ms=None):
    enable_jev_disk_cache(True)
    set_jev_cache_enabled(True)
    set_reuse_jev_cache(True)
    disk = JevDiskCache()
    fused_items = []
    policy_items = []
    e1_rows = []
    query_ms = list(query_embed_ms or [])
    for index, row in enumerate(rows, start=1):
        question = row.get("question") or ""
        if query_embed_ms is None:
            started = time.perf_counter()
            _embed([question])
            query_ms.append((time.perf_counter() - started) * 1000)
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
            jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
        )
        children = fused.get("scored_children") or []
        query = fused.get("query") or question
        scores, _missing = _fill_jev(query, children, disk, live_fill=True)
        children_by_id = {c["chunk_id"]: c for c in children}
        expected = row.get("expected") or []
        qtype = row.get("type") or "factual"
        unans = not bool(row.get("answerable", True))
        present = [s for s in scores.values() if s is not None]
        top_jev = max(present) if present else 0.0
        e1_rows.append(
            {
                "id": row.get("id"),
                "score": -float(top_jev),
                "label": unans,
                "kind": classify_unanswerable(row) if unans else "answerable",
            }
        )
        if not unans:
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
            parents = _parents_from_ids(applied["ordered_ids"], children_by_id, collection_name)
            rank = first_match_rank(parents, expected)
            policy_items.append(
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
    return {
        "fused_items": fused_items,
        "policy_items": policy_items,
        "e1_rows": e1_rows,
        "query_embed_latency_ms": {
            "p50": round(percentile(query_ms, 50), 2),
            "p95": round(percentile(query_ms, 95), 2),
            "mean": round(sum(query_ms) / len(query_ms), 2) if query_ms else 0.0,
            "n": len(query_ms),
            "methodology": "warm timings after at least one prior embed call; excludes cold model load",
        },
    }


def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    def pack(subset):
        if not subset:
            return {"n": 0}
        metrics = {}
        for key, field in (("recall@5", "r5"), ("recall@15", "r15"), ("recall@30", "r30"), ("mrr", "mrr")):
            values = [float(row[field]) for row in subset]
            metrics[key] = round(sum(values) / len(values), 4)
            metrics[f"{key}_ci"] = bootstrap_ci(values, n_resamples=BOOTSTRAP_N)
        metrics["n"] = len(subset)
        return metrics

    return {
        "overall": pack(items),
        "by_type": {t: pack([row for row in items if row.get("type") == t]) for t in TYPES},
    }


def paired_primary(minilm_items, bge_items) -> Dict[str, Any]:
    by_l = {row["id"]: row for row in minilm_items}
    by_r = {row["id"]: row for row in bge_items}
    ids = [i for i in by_l if i in by_r]

    def diff(field, subset_ids=None):
        use = subset_ids if subset_ids is not None else ids
        a = [float(by_l[i][field]) for i in use]
        b = [float(by_r[i][field]) for i in use]
        return {
            "minilm": round(sum(a) / len(a), 4) if a else 0.0,
            "bge": round(sum(b) / len(b), 4) if b else 0.0,
            "diff_ci": bootstrap_diff_ci(b, a, n_resamples=BOOTSTRAP_N),
        }

    exact_ids = [i for i in ids if by_l[i].get("type") == "exact"]
    cells = {
        "policy_B.recall@5": diff("r5"),
        "policy_B.mrr": diff("mrr"),
        "policy_B.exact.recall@5": diff("r5", exact_ids),
        "policy_B.exact.mrr": diff("mrr", exact_ids),
        "policy_B.recall@15": diff("r15"),
        "policy_B.recall@30": diff("r30"),
    }
    return cells


def e1_cv(e1_rows: List[Dict[str, Any]], *, folds: int = 5) -> Dict[str, Any]:
    """5-fold CV threshold for precision at recall>=0.90; report held-out precision."""
    rows = list(e1_rows)
    if len(rows) < folds:
        return {"error": "too few rows", "n": len(rows)}
    rng = random.Random(13)
    order = list(range(len(rows)))
    rng.shuffle(order)
    fold_sizes = [len(rows) // folds] * folds
    for i in range(len(rows) % folds):
        fold_sizes[i] += 1
    folds_idx = []
    cursor = 0
    for size in fold_sizes:
        folds_idx.append(order[cursor : cursor + size])
        cursor += size

    def best_threshold(train):
        scores = [row["score"] for row in train]
        labels = [row["label"] for row in train]
        best = None
        for threshold in sorted(set(scores), reverse=True):
            pred = [score >= threshold for score in scores]
            tp = sum(1 for p, y in zip(pred, labels) if p and y)
            fp = sum(1 for p, y in zip(pred, labels) if p and not y)
            fn = sum(1 for p, y in zip(pred, labels) if (not p) and y)
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            if recall + 1e-12 >= 0.90:
                if best is None or precision > best["precision"]:
                    best = {"threshold": threshold, "precision": precision, "recall": recall}
        return best

    held_precisions = []
    held_by_kind = {"near_miss": [], "off_topic": []}
    for fold in range(folds):
        test_idx = set(folds_idx[fold])
        train = [rows[i] for i in range(len(rows)) if i not in test_idx]
        test = [rows[i] for i in test_idx]
        op = best_threshold(train)
        if not op:
            continue
        thr = op["threshold"]
        pred = [row["score"] >= thr for row in test]
        labels = [row["label"] for row in test]
        tp = sum(1 for p, y in zip(pred, labels) if p and y)
        fp = sum(1 for p, y in zip(pred, labels) if p and not y)
        fn = sum(1 for p, y in zip(pred, labels) if (not p) and y)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        held_precisions.append(precision)
        for kind in ("near_miss", "off_topic"):
            subset = [(p, row) for p, row in zip(pred, test) if row["kind"] == kind]
            # precision among predicted abstains that are this kind of unanswerable vs all pred
            # Report recall of this kind and precision of abstain restricted to kind labels.
            kind_labels = [row["label"] and row["kind"] == kind for row in test]
            # Simpler: among this kind of unanswerable, fraction abstained (recall); and
            # precision of abstain decisions evaluated only on answerable + this kind.
            focus = [row for row in test if (not row["label"]) or row["kind"] == kind]
            if not focus:
                continue
            fpred = [row["score"] >= thr for row in focus]
            flab = [row["label"] for row in focus]
            ftp = sum(1 for p, y in zip(fpred, flab) if p and y)
            ffp = sum(1 for p, y in zip(fpred, flab) if p and not y)
            held_by_kind[kind].append(ftp / (ftp + ffp) if (ftp + ffp) else 0.0)

    scores = [row["score"] for row in rows]
    labels = [row["label"] for row in rows]
    return {
        "folds": folds,
        "auroc": auroc(scores, labels),
        "heldout_precision_at_recall_ge_0.90_mean": round(
            sum(held_precisions) / len(held_precisions), 4
        )
        if held_precisions
        else None,
        "heldout_precision_folds": [round(value, 4) for value in held_precisions],
        "heldout_precision_by_kind": {
            kind: round(sum(values) / len(values), 4) if values else None
            for kind, values in held_by_kind.items()
        },
        "kind_counts": dict(Counter(row["kind"] for row in rows if row["label"])),
    }


def apply_decision_rule(cells: Dict[str, Any]) -> Dict[str, Any]:
    primary_keys = [
        "policy_B.recall@5",
        "policy_B.mrr",
        "policy_B.exact.recall@5",
        "policy_B.exact.mrr",
    ]
    non_negative = True
    any_improved = False
    details = {}
    for key in primary_keys:
        ci = (cells.get(key) or {}).get("diff_ci") or {}
        low = float(ci.get("low") or 0.0)
        details[key] = {"low": low, "high": ci.get("high"), "diff": ci.get("diff")}
        if low <= -0.01:
            non_negative = False
        if low > 0:
            any_improved = True
    exact_r5_low = float(((cells.get("policy_B.exact.recall@5") or {}).get("diff_ci") or {}).get("low") or 0.0)
    exact_ok = exact_r5_low > -0.01
    switch = bool(non_negative and any_improved and exact_ok)
    return {
        "switch_to_bge_m3": switch,
        "non_negative_primaries": non_negative,
        "any_primary_improved_outside_zero": any_improved,
        "exact_recall@5_not_regressed": exact_ok,
        "primary_cells_compared": len(primary_keys),
        "all_cells_compared": len(cells),
        "multiple_comparison_risk": DECISION_RULE["multiple_comparison_note"],
        "details": details,
        "verdict": (
            "SWITCH to BGE-M3"
            if switch
            else "KEEP MiniLM (or run a bigger test)"
        ),
    }


def measure_query_latency(client, questions: Sequence[str], *, warm_n: int = 3) -> Dict[str, Any]:
    # Cold: first call after client creation may include connection setup.
    cold = []
    if questions:
        started = time.perf_counter()
        client.embed([questions[0]])
        cold.append((time.perf_counter() - started) * 1000)
    # Warm-up discarded calls.
    for question in questions[1 : 1 + warm_n]:
        client.embed([question])
    warm = []
    for question in questions[1 + warm_n :]:
        started = time.perf_counter()
        client.embed([question])
        warm.append((time.perf_counter() - started) * 1000)
    return {
        "cold_first_ms": round(cold[0], 2) if cold else None,
        "warm_p50_ms": round(percentile(warm, 50), 2) if warm else None,
        "warm_p95_ms": round(percentile(warm, 95), 2) if warm else None,
        "warm_n": len(warm),
        "methodology": (
            "Cold = first embed after client init. Warm = subsequent single-query embeds; "
            f"discarded {warm_n} warm-up queries before measuring."
        ),
    }


def index_size(chunk_count: int, dims: int) -> Dict[str, Any]:
    theoretical = chunk_count * dims * 4
    return {
        "chunk_count": chunk_count,
        "dims": dims,
        "theoretical_dense_bytes": theoretical,
        "theoretical_dense_mb": round(theoretical / (1024 * 1024), 3),
    }


def _print_summary_table(title: str, summary: Dict[str, Any]) -> None:
    print(f"\n### {title}")
    print("| split | n | r@5 | r@15 | r@30 | MRR |")
    print("|---|---:|---:|---:|---:|---:|")
    overall = summary.get("overall") or {}
    if overall.get("n"):
        print(
            f"| overall | {overall['n']} | {overall['recall@5']:.4f} | "
            f"{overall['recall@15']:.4f} | {overall['recall@30']:.4f} | {overall['mrr']:.4f} |"
        )
    for qtype in TYPES:
        block = (summary.get("by_type") or {}).get(qtype) or {}
        if not block.get("n"):
            continue
        print(
            f"| {qtype} | {block['n']} | {block['recall@5']:.4f} | "
            f"{block['recall@15']:.4f} | {block['recall@30']:.4f} | {block['mrr']:.4f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--golden", default=GOLDEN)
    parser.add_argument("--keep-indexes", action="store_true")
    parser.add_argument("--skip-openrouter-safe", action="store_true", help="Embed safe docs via TEI too.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    os.environ.setdefault("TEI_URL", "http://127.0.0.1:18080")

    config = load_config(args.config)
    rows = load_jsonl(args.golden)
    inventory = corpus_inventory()
    coverage = golden_coverage(rows)

    report: Dict[str, Any] = {
        "phase": "0.9",
        "decision_rule_predeclared": DECISION_RULE,
        "corpus": inventory,
        "golden_coverage": coverage,
        "production_defaults_changed": False,
        "published": False,
    }
    # Persist the pre-declared rule BEFORE running metrics.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump({"phase": "0.9", "decision_rule_predeclared": DECISION_RULE, "status": "started"}, handle, indent=2)
        handle.write("\n")
    print("Pre-declared decision rule written to report.", flush=True)
    print(json.dumps(DECISION_RULE, indent=2), flush=True)
    print("Corpus sensitivity flag:", inventory["flag"], flush=True)
    print(json.dumps({"corpus": inventory, "coverage": coverage}, indent=2), flush=True)

    if not coverage["meets_150_answerable"] or not coverage["meets_5_documents"]:
        print(
            "WARNING: golden coverage gate not met "
            f"(answerable={coverage['n_answerable']}, docs={coverage['documents_with_answerable']}). "
            "Continue after expansion if possible.",
            file=sys.stderr,
            flush=True,
        )

    tei = TeiEmbeddingClient()
    if not tei.health():
        raise SystemExit(
            "TEI backend not healthy at TEI_URL. Start docker compose -f docker-compose.tei.yml "
            "or python -m embeddings.tei_local_server"
        )

    openrouter = None if args.skip_openrouter_safe else BgeM3EmbeddingClient()
    router = RoutingBgeM3Client(tei=tei, openrouter=openrouter, default_backend="tei")

    # Latency probes (MiniLM / TEI / OpenRouter) before long index build.
    questions = [row.get("question") or "" for row in rows if row.get("question")][:40]
    from embeddings import MiniLMEmbeddingClient

    minilm_client = MiniLMEmbeddingClient()
    # Warm MiniLM model load separately.
    minilm_client.embed(["warmup"])
    latency_block = {
        "minilm_warm": measure_query_latency(minilm_client, questions),
        "bge_tei_cpu": measure_query_latency(tei, questions),
        "bge_openrouter": measure_query_latency(openrouter, questions) if openrouter else None,
        "bge_tei_gpu": None,
        "gpu_available": False,
    }
    report["latency"] = latency_block

    # Cross-backend cosine on up to 200 safe chunks if both backends used.
    safe_texts = [row["text"] for row in _chunk_rows(config.get("embed_style") or "raw") if row["backend"] == "openrouter"]
    sample = safe_texts[:200]
    if openrouter and sample:
        report["cross_backend_cosine"] = assert_cross_backend_cosine(tei, openrouter, sample, min_cosine=0.99)
    else:
        report["cross_backend_cosine"] = {"skipped": True, "reason": "single backend or no safe sample"}

    # Cost estimates
    all_texts = [row["text"] for row in _chunk_rows(config.get("embed_style") or "raw")]
    safe_n = sum(1 for row in _chunk_rows(config.get("embed_style") or "raw") if row["backend"] == "openrouter")
    sens_n = len(all_texts) - safe_n
    openrouter_est = openrouter.estimate_cost_usd(sample and safe_texts or []) if openrouter else 0.0
    # Re-estimate on all safe texts
    if openrouter:
        openrouter_est = openrouter.estimate_cost_usd(
            [row["text"] for row in _chunk_rows(config.get("embed_style") or "raw") if row["backend"] == "openrouter"]
        )
    estimate = {
        "jev": round(len(rows) * int(config.get("top_k", 30)) * _JEV_COST_PER_CANDIDATE, 4),
        "writer": 0.0,
        "checker": 0.0,
        "rewriter": 0.0,
        "total": round((openrouter_est or 0) + len(rows) * int(config.get("top_k", 30)) * _JEV_COST_PER_CANDIDATE, 4),
        "openrouter_safe_reembed_usd": round(openrouter_est or 0, 6),
        "tei_sensitive_chunks": sens_n,
        "tei_safe_chunks_if_routed": safe_n,
        "tei_cost_usd": 0.0,
        "batch_size_tei": tei.batch_size,
    }
    report["cost_estimate"] = estimate
    print(
        f"Estimated OpenRouter safe re-embed ${estimate['openrouter_safe_reembed_usd']} "
        f"for {safe_n} chunks; TEI embeds {sens_n} sensitive chunks locally (batch={tei.batch_size})",
        flush=True,
    )
    assert_budget_for_estimate(estimate, label="phase09")

    built = []
    try:
        set_embedding_client_override(None)
        print("Building MiniLM full-corpus throwaway...", file=sys.stderr, flush=True)
        minilm_meta = build_throwaway_index(
            embed_style=config.get("embed_style") or "raw",
            chunking=config.get("chunking") or "Parent-child",
            embedding_model_id=EMBEDDING_MODEL_ID,
            force_reembed=False,
        )
        minilm_meta["document_count"] = inventory["document_count"]
        built.append(minilm_meta)

        print("Building BGE-M3 full-corpus throwaway (routed)...", file=sys.stderr, flush=True)
        if args.skip_openrouter_safe:
            # Force all TEI
            for row in _chunk_rows(config.get("embed_style") or "raw"):
                row["backend"] = "tei"
        bge_meta, _extra = build_bge_full_index(
            embed_style=config.get("embed_style") or "raw",
            chunking=config.get("chunking") or "Parent-child",
            router=router,
        )
        built.append(bge_meta)

        tei_cps = None
        if bge_meta.get("corpus_embed_ms") and inventory["chunk_count"]:
            # Approximate TEI throughput from TEI-only portion if available.
            tei_ms = float(tei.stats.get("embed_ms") or 0)
            tei_chunks = int(tei.stats.get("chunks") or 0)
            if tei_ms > 0 and tei_chunks:
                tei_cps = round(tei_chunks / (tei_ms / 1000.0), 2)

        set_embedding_client_override(None)
        minilm_eval = evaluate_index(rows, config, collection_name=minilm_meta["collection_name"], label="minilm")

        override = RoutedOverride(router, query_backend="tei")
        set_embedding_client_override(override)
        # Warm TEI query path once
        _embed(["warmup query"])
        override.query_latencies_ms.clear()
        bge_eval = evaluate_index(rows, config, collection_name=bge_meta["collection_name"], label="bge")
        bge_eval["query_embed_latency_ms"] = {
            "p50": round(percentile(override.query_latencies_ms, 50), 2) if override.query_latencies_ms else 0,
            "p95": round(percentile(override.query_latencies_ms, 95), 2) if override.query_latencies_ms else 0,
            "mean": round(sum(override.query_latencies_ms) / len(override.query_latencies_ms), 2)
            if override.query_latencies_ms
            else 0,
            "n": len(override.query_latencies_ms),
            "methodology": "warm TEI query embeds during eval loop",
        }

        cells = paired_primary(minilm_eval["policy_items"], bge_eval["policy_items"])
        # Add fused paired diffs as secondary cells
        fused_cells = paired_primary(minilm_eval["fused_items"], bge_eval["fused_items"])
        for key, value in fused_cells.items():
            cells[f"fused.{key.split('.', 1)[-1] if key.startswith('policy_B.') else key}"] = value
        # Fix fused keys cleanly
        cells = paired_primary(minilm_eval["policy_items"], bge_eval["policy_items"])
        for key, value in paired_primary(minilm_eval["fused_items"], bge_eval["fused_items"]).items():
            cells["fused_only." + key.split("policy_B.", 1)[-1]] = value

        decision = apply_decision_rule(cells)
        e1_minilm = e1_cv(minilm_eval["e1_rows"])
        e1_bge = e1_cv(bge_eval["e1_rows"])

        report.update(
            {
                "minilm_index": minilm_meta,
                "bge_index": bge_meta,
                "minilm_fused": summarize(minilm_eval["fused_items"]),
                "bge_fused": summarize(bge_eval["fused_items"]),
                "minilm_policy_B": summarize(minilm_eval["policy_items"]),
                "bge_policy_B": summarize(bge_eval["policy_items"]),
                "paired_diff_cis": cells,
                "e1_minilm": e1_minilm,
                "e1_bge": e1_bge,
                "decision": decision,
                "tei_throughput_chunks_per_s": tei_cps,
                "index_size_bge": index_size(int(bge_meta.get("chunk_count") or 0), 1024),
                "index_size_minilm": index_size(int(minilm_meta.get("chunk_count") or 0), 384),
                "query_latency_eval": {
                    "minilm": minilm_eval["query_embed_latency_ms"],
                    "bge_tei": bge_eval["query_embed_latency_ms"],
                },
                "open_items": [
                    "Docker Desktop install may still be pending; local TEI shim used if compose unavailable.",
                    "GPU TEI profile not measured if no NVIDIA device.",
                    "Family-wise error across primary metric cells is uncontrolled.",
                    "Schema columns embedding_model/embedding_dim still design-only.",
                ],
                "remaining_budget_after": openrouter_remaining_budget(),
            }
        )
        with open(OUT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

        # Tables
        print("\n## Corpus")
        print(f"chunks={inventory['chunk_count']} documents={inventory['document_count']}")
        print(json.dumps(inventory["chunks_by_document"], indent=2))
        print(json.dumps(inventory["sensitivity"], indent=2))
        print(f"Backend routing: {bge_meta.get('backend_chunk_counts')}")

        _print_summary_table("MiniLM fused-only", report["minilm_fused"])
        _print_summary_table("BGE-M3 fused-only", report["bge_fused"])
        _print_summary_table("MiniLM policy B", report["minilm_policy_B"])
        _print_summary_table("BGE-M3 policy B", report["bge_policy_B"])

        print("\n### Paired differences (BGE - MiniLM), 2000× bootstrap 95% CI")
        print("| metric | MiniLM | BGE | diff | CI low | CI high |")
        print("|---|---:|---:|---:|---:|---:|")
        for key, block in cells.items():
            ci = block.get("diff_ci") or {}
            print(
                f"| {key} | {block['minilm']:.4f} | {block['bge']:.4f} | "
                f"{ci.get('diff')} | {ci.get('low')} | {ci.get('high')} |"
            )

        print("\n### Abstain E1 (5-fold CV held-out precision @ recall>=0.90)")
        print("| embedder | AUROC | held-out P | near_miss P | off_topic P |")
        print("|---|---:|---:|---:|---:|")
        for name, block in (("MiniLM", e1_minilm), ("BGE-M3", e1_bge)):
            kind = block.get("heldout_precision_by_kind") or {}
            print(
                f"| {name} | {block.get('auroc')} | {block.get('heldout_precision_at_recall_ge_0.90_mean')} | "
                f"{kind.get('near_miss')} | {kind.get('off_topic')} |"
            )

        print("\n### Latency / cost / size")
        print(json.dumps({
            "latency": latency_block,
            "tei_throughput_chunks_per_s": tei_cps,
            "batch_size_tei": tei.batch_size,
            "openrouter_safe_reembed_usd": estimate["openrouter_safe_reembed_usd"],
            "index_size_bge_mb": report["index_size_bge"]["theoretical_dense_mb"],
            "cross_backend_cosine": report.get("cross_backend_cosine"),
        }, indent=2))

        print("\n### Verdict")
        print(decision["verdict"])
        print(json.dumps(decision, indent=2))
        print("\nOpen items:")
        for item in report["open_items"]:
            print(f"- {item}")
        print(f"\nWrote {OUT_PATH}")
        return 0
    finally:
        set_embedding_client_override(None)
        if not args.keep_indexes:
            for meta in built:
                try:
                    discard_index_version(meta["version_id"], meta["collection_name"])
                    print(f"Discarded {meta['collection_name']}", file=sys.stderr, flush=True)
                except Exception as exc:
                    print(f"Discard failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
