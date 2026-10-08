"""Run golden questions through the production pipeline and score each one.

Import this only after NEEDLE_DATA_DIR points at the index to evaluate: it imports the engine modules.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from catalog import active_collection, document_hash_map, store
from pipeline import PipelineOptions, answer_question
from versions import available_documents as indexed_documents
from eval import metrics as m
from eval.schema import documents_covered, parent_matches

RANK_DEPTH = 10


def pipeline_options(config: Dict[str, Any], tier: Dict[str, Any]) -> PipelineOptions:
    pipeline = config["pipeline"]
    return PipelineOptions(
        top_k=int(pipeline["top_k"]),
        similarity_threshold=float(pipeline["similarity_threshold"]),
        rrf_k=int(pipeline["rrf_k"]),
        max_parents=int(pipeline["max_parents"]),
        jev_relevance_threshold=float(pipeline["jev_relevance_threshold"]),
        retry_threshold=float(pipeline["retry_threshold"]),
        retry_top_k=int(pipeline["retry_top_k"]),
        jev_candidate_limit=int(pipeline.get("jev_candidate_limit") or 0),
        answer_length=pipeline["answer_length"],
        require_citations=bool(pipeline["require_citations"]),
        citation_style=pipeline["citation_style"],
        withhold_ungrounded=True,
        rerank_mode=tier["rerank_mode"],
        allow_retry=bool(tier["allow_retry"]),
        generate=bool(tier["generate"]),
        collect_ranking=True,
    )


def available_documents() -> Dict[str, set]:
    _collection, active = active_collection()
    return indexed_documents(active["collection_name"])


def _hash_map() -> Dict[str, str]:
    _collection, active = active_collection()
    return document_hash_map(active["collection_name"])


def score_row(row: Dict[str, Any], result: Dict[str, Any], *, k: int, generate: bool) -> Dict[str, Any]:
    """Turn one pipeline result into a flat, JSON-safe record with per-question scores."""
    expected = row.get("expected") or []
    retrieval = result.get("retrieval") or {}
    full_ranking = list(retrieval.get("ranked") or [])
    ranked = full_ranking[:RANK_DEPTH]
    deep_ranks = [
        rank for rank, parent in enumerate(full_ranking, start=1)
        if any(parent_matches(parent, item) for item in expected)
    ]
    match_ranks = [
        rank for rank, parent in enumerate(ranked, start=1)
        if any(parent_matches(parent, item) for item in expected)
    ]
    # nDCG counts each expected passage once, at the first rank that contains it.
    item_ranks = sorted({
        next(rank for rank, parent in enumerate(ranked, start=1) if parent_matches(parent, item))
        for item in expected
        if any(parent_matches(parent, item) for parent in ranked)
    })
    first_rank = match_ranks[0] if match_ranks else None
    found_at_k = [any(parent_matches(parent, item) for parent in ranked[:k]) for item in expected]
    page_items = [item for item in expected if item.get("page")]
    page_hit = None
    if page_items:
        page_hit = any(
            parent_matches(parent, item) and int(parent.get("page_number") or 0) == int(item["page"])
            for parent in ranked[:k]
            for item in page_items
        )

    record: Dict[str, Any] = {
        "id": row["id"],
        "type": row.get("type"),
        "tags": row.get("tags") or [],
        "answerable": bool(row.get("answerable", True)),
        "question": row.get("question"),
        "asked": result.get("question"),
        "search_query": retrieval.get("search_query"),
        "rerank_mode": retrieval.get("mode"),
        "top_relevance": retrieval.get("relevance"),
        "retry_used": retrieval.get("retry_used"),
        "ranked": [
            {
                "document": parent.get("document_name"),
                "page": parent.get("page_number"),
                "section": parent.get("header_context"),
                "score": parent.get("similarity_score"),
                "match": rank in match_ranks,
                "snippet": (parent.get("text") or "")[:160],
            }
            for rank, parent in enumerate(ranked, start=1)
        ],
        "latencies_ms": result.get("latencies_ms") or {},
        "cost_usd": result.get("cost_usd") or 0.0,
        "infra_error": bool(result.get("infra_error")),
        "error": result.get("error"),
        "reject_category": result.get("reject_category"),
    }
    if record["answerable"] and expected:
        record["scores"] = {
            "hit_at_1": m.hit_at_k(first_rank, 1),
            "hit_at_3": m.hit_at_k(first_rank, 3),
            "hit_at_5": m.hit_at_k(first_rank, 5),
            "hit_at_10": m.hit_at_k(first_rank, 10),
            "recall_at_k": sum(found_at_k) / len(found_at_k),
            "rr": m.reciprocal_rank(first_rank),
            "ndcg_at_k": round(m.ndcg_at_k(item_ranks, len(expected), k), 4),
        }
        if page_hit is not None:
            record["scores"]["page_hit_at_k"] = 1.0 if page_hit else 0.0
        record["first_match_rank"] = first_rank
        # Rank of the evidence anywhere in the candidate list (up to 30); sizes JEV_CANDIDATE_LIMIT.
        record["deep_match_rank"] = deep_ranks[0] if deep_ranks else None
    else:
        record["scores"] = {}

    if not generate:
        return record

    answer = result.get("answer") or ""
    released = bool(result.get("released"))
    sources = result.get("sources") or []
    citations = m.citation_numbers(answer) if released else []
    record.update(
        {
            "answer": answer,
            "draft": result.get("draft") if not released else None,
            "released": released,
            "passed": bool(result.get("passed")),
            "declined": bool(result.get("declined")),
            "abstained": not released or bool(result.get("declined")),
            "grounded": result.get("grounded"),
            "partially_supported": bool(result.get("partially_supported")),
            "reason": result.get("reason") or "",
            "citations": citations,
            "raw_checker": ((result.get("verdict") or {}).get("raw_checker") if not result.get("passed") else None),
        }
    )
    scores = record["scores"]
    leaks = m.leaked_phrases(answer, row.get("forbidden_phrases") or []) if released else []
    record["leaked_phrases"] = leaks
    if record["answerable"]:
        scores["answered"] = 1.0 if released and result.get("passed") and not result.get("declined") else 0.0
        if released and not result.get("declined"):
            facts = m.key_fact_recall(answer, row.get("key_facts") or [])
            if facts is not None:
                scores["key_fact_recall"] = round(facts, 4)
                scores["fully_correct"] = 1.0 if facts == 1.0 else 0.0
            scores["citation_valid"] = 1.0 if citations and all(1 <= n <= len(sources) for n in citations) else 0.0
            cited = [sources[n - 1] for n in citations if 1 <= n <= len(sources)]
            scores["cites_evidence"] = 1.0 if any(parent_matches(src, item) for src in cited for item in expected) else 0.0
    return record


def run_rows(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    tier: Dict[str, Any],
    *,
    workers: int = 1,
    max_cost: Optional[float] = None,
    judge: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    options = pipeline_options(config, tier)
    k = int(config["k"])
    hashes = _hash_map()
    spent = 0.0
    lock = threading.Lock()
    stop = threading.Event()
    records: Dict[int, Dict[str, Any]] = {}

    def one(index: int, row: Dict[str, Any]) -> None:
        nonlocal spent
        if stop.is_set():
            return
        use_history = bool(tier["use_history"]) and bool(row.get("history"))
        question = row["question"] if use_history or not row.get("history") else row.get("standalone_question") or row["question"]
        started = time.perf_counter()
        result = answer_question(
            question,
            history=row.get("history") if use_history else None,
            options=options,
        )
        for parent in (result.get("retrieval") or {}).get("ranked") or []:
            parent["content_hash"] = hashes.get(parent.get("document_id") or "", "")
        for source in result.get("sources") or []:
            source["content_hash"] = hashes.get(source.get("document_id") or "", "")
        record = score_row(row, result, k=k, generate=bool(tier["generate"]))
        record["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if judge and record.get("released") and record["answerable"]:
            verdict = judge(row, record)
            record["judge"] = verdict
            record["cost_usd"] = round(record["cost_usd"] + float(verdict.get("cost_usd") or 0), 6)
            if verdict.get("score") is not None:
                record["scores"]["judge_correct"] = float(verdict["score"])
        with lock:
            records[index] = record
            spent += float(record["cost_usd"] or 0)
            done = len(records)
            if max_cost is not None and spent > max_cost:
                stop.set()
        if progress and (done == 1 or done % 10 == 0 or done == len(rows)):
            print(f"  {done}/{len(rows)} questions · ${spent:.4f} spent", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, index, row) for index, row in enumerate(rows)]
        for future in as_completed(futures):
            future.result()

    ordered = [records[index] for index in sorted(records)]
    return {
        "records": ordered,
        "complete": len(ordered) == len(rows),
        "stopped_for_budget": stop.is_set(),
        "spent_usd": round(spent, 6),
    }


def split_covered(rows: List[Dict[str, Any]]) -> tuple:
    """(rows whose documents are in the index, ids of rows that cannot be scored here)."""
    available = available_documents()
    covered, uncovered = [], []
    for row in rows:
        (covered if documents_covered(row, available) else uncovered).append(row)
    return covered, [row["id"] for row in uncovered]


def unreachable_spans(rows: List[Dict[str, Any]]) -> List[str]:
    """Expected spans that are not inside any single parent passage of their document.

    A span split across a chunk boundary can never be matched, so it is a dataset bug.
    """
    parents = store.conn.execute("SELECT document_name, page_number, parent_text FROM parents").fetchall()
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for row in parents:
        by_doc.setdefault(row["document_name"], []).append(
            {"document_name": row["document_name"], "page_number": row["page_number"], "text": row["parent_text"]}
        )
    problems = []
    for row in rows:
        for item in row.get("expected") or []:
            name = item.get("document")
            candidates = by_doc.get(name) or []
            hits = [parent for parent in candidates if parent_matches(parent, item)]
            if not hits:
                problems.append(f"{row['id']}: span not found in any parent of {name!r}: {item.get('answer_span')!r}")
            elif item.get("page") and not any(int(p["page_number"]) == int(item["page"]) for p in hits):
                pages = sorted({p["page_number"] for p in hits})
                problems.append(f"{row['id']}: span is on page(s) {pages} of {name!r}, not page {item['page']}")
    return problems
