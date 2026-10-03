"""Retrieval miss analysis for golden items with recall@5 = 0.

No writer/checker LLM calls. Fused ranks are free; Jev ranks use the reranker only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.metrics import hit_from_rank, mrr_from_rank
from eval.schema import first_match_rank, load_jsonl, write_jsonl
from pipeline_logic import corpus_contains_span
from rag_engine import (
    RETRY_THRESHOLD,
    list_parent_passages,
    retrieve_parents,
)


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "baseline_config.json")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")
TYPES = ("factual", "exact", "multihop", "followup", "unanswerable")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _label_in_index(
    expected: Sequence[Dict[str, str]],
    passages: Sequence[Dict[str, Any]],
    chunk_texts: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Whether content_hash+span exists in any stored parent or child chunk."""
    for item in expected:
        content_hash = item.get("content_hash") or ""
        span = item.get("answer_span") or ""
        if not content_hash:
            continue
        for parent in passages:
            if parent.get("content_hash") != content_hash:
                continue
            text = parent.get("text") or ""
            if not span or corpus_contains_span(text, span):
                return {
                    "valid": True,
                    "location": "parent",
                    "parent_id": parent.get("parent_id"),
                    "content_hash": content_hash,
                }
        for chunk_hash, text in chunk_texts:
            if chunk_hash != content_hash:
                continue
            if not span or corpus_contains_span(text, span):
                return {
                    "valid": True,
                    "location": "chunk",
                    "content_hash": content_hash,
                }
    return {"valid": False, "location": None, "content_hash": (expected[0].get("content_hash") if expected else None)}


def _chunk_texts_by_hash() -> List[Tuple[str, str]]:
    from rag_engine import _active_collection, _document_hash_map, chroma_client

    collection, active = _active_collection()
    hash_map = _document_hash_map(active["collection_name"])
    payload = collection.get(include=["metadatas", "documents"])
    rows: List[Tuple[str, str]] = []
    for text, metadata in zip(payload.get("documents") or [], payload.get("metadatas") or []):
        metadata = metadata or {}
        document_id = metadata.get("document_id") or ""
        content_hash = hash_map.get(document_id, "")
        if content_hash:
            rows.append((content_hash, text or ""))
    return rows


def _retrieve(
    row: Dict[str, Any],
    config: Dict[str, Any],
    *,
    use_jev: bool,
) -> Dict[str, Any]:
    return retrieve_parents(
        row.get("question") or "",
        history=row.get("history") or [],
        top_k=int(config.get("top_k", 30)),
        similarity_threshold=float(config.get("similarity_threshold", 0.30)),
        rrf_k=int(config.get("rrf_k", 60)),
        max_parents=max(int(config.get("max_parents", 5)), 30),
        use_jev=use_jev,
        jev_relevance_threshold=float(config.get("jev_relevance_threshold", 0.20)),
        retry_threshold=float(config.get("retry_threshold", RETRY_THRESHOLD)),
        allow_retry=False,
    )


def _rank_in_list(parents: Sequence[Dict[str, Any]], expected: Sequence[Dict[str, str]]) -> Optional[int]:
    return first_match_rank(parents, expected)


def _jev_details(result: Dict[str, Any], expected: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    ordered = result.get("ordered_parents") or result.get("parents") or []
    rank = _rank_in_list(ordered, expected)
    reached = False
    jev_score = None
    if rank is not None and 1 <= rank <= len(ordered):
        hit = ordered[rank - 1]
        reached = bool(hit.get("reached_jev"))
        jev_score = hit.get("jev_score")
    else:
        for parent in ordered:
            # Expected may sit in the fused tail appended after Jev.
            from eval.schema import parent_matches_expected

            if parent_matches_expected(parent, expected):
                reached = bool(parent.get("reached_jev"))
                jev_score = parent.get("jev_score")
                break
        for child in result.get("scored_children") or []:
            # Child-level reach when parent list was truncated.
            if child.get("reached_jev") and jev_score is None:
                pass
    return {"rank": rank, "reached_jev": reached, "jev_score": jev_score}


def _type_recalls(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    use_jev: bool,
) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"r5": [], "r15": [], "r30": [], "mrr": []})
    overall = {"r5": [], "r15": [], "r30": [], "mrr": []}
    for index, row in enumerate(rows, start=1):
        qtype = row.get("type") or "factual"
        if qtype == "unanswerable" or not row.get("answerable", True):
            continue
        result = _retrieve(row, config, use_jev=use_jev)
        parents = (
            (result.get("ordered_parents") if use_jev else result.get("fused_parents"))
            or result.get("parents")
            or []
        )
        if use_jev and not parents:
            parents = result.get("parents") or []
        if not use_jev:
            parents = result.get("fused_parents") or result.get("ordered_parents") or result.get("parents") or []
        rank = _rank_in_list(parents, row.get("expected") or [])
        for key, k in (("r5", 5), ("r15", 15), ("r30", 30)):
            value = hit_from_rank(rank, k)
            by_type[qtype][key].append(value)
            overall[key].append(value)
        mrr = mrr_from_rank(rank)
        by_type[qtype]["mrr"].append(mrr)
        overall["mrr"].append(mrr)
        if index == 1 or index % 10 == 0 or index == len(rows):
            mode = "jev" if use_jev else "fused"
            print(f"{mode} recall {index}/{len(rows)}", file=sys.stderr, flush=True)

    def _avg(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 4) if values else None

    table = {"overall": {key: _avg(values) for key, values in overall.items()}, "by_type": {}}
    for qtype in TYPES:
        if qtype == "unanswerable":
            table["by_type"][qtype] = {"recall@5": "n/a", "recall@15": "n/a", "recall@30": "n/a", "mrr": "n/a", "n": 0}
            continue
        bucket = by_type.get(qtype) or {"r5": [], "r15": [], "r30": [], "mrr": []}
        table["by_type"][qtype] = {
            "recall@5": _avg(bucket["r5"]),
            "recall@15": _avg(bucket["r15"]),
            "recall@30": _avg(bucket["r30"]),
            "mrr": _avg(bucket["mrr"]),
            "n": len(bucket["r5"]),
        }
    return table


def analyze_misses(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    passages = list_parent_passages()
    chunk_texts = _chunk_texts_by_hash()
    answerable = [row for row in rows if row.get("answerable", True)]

    print("Computing fused recalls (no Jev)…", file=sys.stderr, flush=True)
    fused_table = _type_recalls(answerable, config, use_jev=False)
    print("Computing Jev recalls…", file=sys.stderr, flush=True)
    jev_table = _type_recalls(answerable, config, use_jev=True)

    misses: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for index, row in enumerate(answerable, start=1):
        # Misses are defined on the production path (Jev recall@5 = 0).
        jev = _retrieve(row, config, use_jev=True)
        jev_info = _jev_details(jev, row.get("expected") or [])
        jev_rank = jev_info["rank"]
        if hit_from_rank(jev_rank, 5) == 1.0:
            if index == 1 or index % 10 == 0 or index == len(answerable):
                print(f"scan {index}/{len(answerable)}", file=sys.stderr, flush=True)
            continue
        fused = _retrieve(row, config, use_jev=False)
        fused_parents = fused.get("fused_parents") or fused.get("ordered_parents") or fused.get("parents") or []
        fused_rank = _rank_in_list(fused_parents, row.get("expected") or [])
        # Prefer fused_rank from the fused_parents attached to the Jev call when present.
        fused_from_jev = _rank_in_list(jev.get("fused_parents") or [], row.get("expected") or [])
        if fused_from_jev is not None:
            fused_rank = fused_from_jev
        label = _label_in_index(row.get("expected") or [], passages, chunk_texts)
        item = {
            "id": row.get("id"),
            "type": row.get("type"),
            "question": row.get("question"),
            "label_valid": label["valid"],
            "label_location": label.get("location"),
            "fused_rank_in_30": fused_rank,
            "jev_rank": jev_rank,
            "reached_jev": jev_info["reached_jev"],
            "jev_score": jev_info["jev_score"],
            "expected": row.get("expected"),
        }
        misses.append(item)
        if not label["valid"]:
            invalid.append(item)
        print(f"miss {len(misses)} at item {index}/{len(answerable)} id={row.get('id')}", file=sys.stderr, flush=True)

    proposals = _proposals(misses)
    return {
        "answerable": len(answerable),
        "recall@5_misses": len(misses),
        "invalid_labels": invalid,
        "misses": misses,
        "fused": fused_table,
        "jev": jev_table,
        "proposed_retrieval_changes": proposals,
        "applied": False,
    }


def _proposals(misses: List[Dict[str, Any]]) -> List[str]:
    proposals: List[str] = []
    real = [item for item in misses if item.get("label_valid")]
    if not real:
        return ["No real retrieval misses after invalid-label filtering."]
    deep = [item for item in real if item.get("fused_rank_in_30") and item["fused_rank_in_30"] > 5]
    absent = [item for item in real if item.get("fused_rank_in_30") is None]
    not_jev = [
        item
        for item in real
        if item.get("fused_rank_in_30") and item.get("reached_jev") is False
    ]
    if absent:
        proposals.append(
            f"{len(absent)} valid labels never appear in the fused top-30; "
            "consider widening keyword OR-query coverage / synonym expansion before RRF, "
            "or lowering the vector similarity floor for rare exact terms."
        )
    if deep:
        proposals.append(
            f"{len(deep)} valid labels sit in fused ranks 6–30; "
            "consider sending more candidates to Jev (e.g. 20–30) or a small lexical boost "
            "when the expected span tokens overlap the query."
        )
    if not_jev:
        proposals.append(
            f"{len(not_jev)} hits were in the fused list but never reached Jev; "
            "raise jev_candidate_limit so mid-fused parents are scored."
        )
    low = [item for item in real if item.get("reached_jev") and (item.get("jev_score") or 0) < 0.2]
    if low:
        proposals.append(
            f"{len(low)} reached Jev but scored below the relevance floor; "
            "do not lower jev_relevance_threshold yet—inspect whether parent text is too broad "
            "versus the answer span (parent-child split / header context)."
        )
    if not proposals:
        proposals.append("Real misses are mixed; inspect per-item ranks before changing defaults.")
    return proposals


def fix_invalid_labels(rows: List[Dict[str, Any]], invalid_ids: Sequence[str], passages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Repair answer_span / content_hash when the span is nearby in the same hash family."""
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for parent in passages:
        by_hash[parent.get("content_hash") or ""].append(parent)

    fixed = []
    changed_rows = []
    invalid_set = set(invalid_ids)
    for row in rows:
        if row.get("id") not in invalid_set or not row.get("answerable", True):
            changed_rows.append(row)
            continue
        expected = list(row.get("expected") or [])
        repaired = False
        new_expected = []
        for item in expected:
            span = (item.get("answer_span") or "").strip()
            content_hash = item.get("content_hash") or ""
            # Prefer same hash with normalized/nearby span; else search all parents for the span.
            candidates = by_hash.get(content_hash) or []
            chosen = None
            for parent in candidates:
                if span and corpus_contains_span(parent.get("text") or "", span):
                    chosen = parent
                    break
            if chosen is None and span:
                for parent in passages:
                    if corpus_contains_span(parent.get("text") or "", span):
                        chosen = parent
                        break
            if chosen is None and span:
                # Truncate/expand span to a contiguous window present in a same-hash parent.
                for parent in candidates or passages:
                    text = parent.get("text") or ""
                    tokens = span.split()
                    for width in range(min(12, len(tokens)), 3, -1):
                        needle = " ".join(tokens[:width])
                        if corpus_contains_span(text, needle):
                            new_expected.append(
                                {
                                    "content_hash": parent.get("content_hash") or content_hash,
                                    "answer_span": needle,
                                }
                            )
                            repaired = True
                            chosen = parent
                            break
                    if chosen is not None:
                        break
            if chosen is not None and not repaired:
                new_expected.append(
                    {
                        "content_hash": chosen.get("content_hash") or content_hash,
                        "answer_span": span,
                    }
                )
                if (chosen.get("content_hash") or content_hash) != content_hash:
                    repaired = True
            elif chosen is None:
                new_expected.append(item)
        if repaired or new_expected != expected:
            updated = dict(row)
            updated["expected"] = new_expected
            changed_rows.append(updated)
            fixed.append({"id": row.get("id"), "before": expected, "after": new_expected})
        else:
            changed_rows.append(row)
    return changed_rows, fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze golden recall@5 misses (no writer/checker).")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--golden", default=GOLDEN)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "baselines", "miss_analysis.json"))
    parser.add_argument("--fix-invalid", action="store_true", help="Rewrite golden.jsonl spans that are not in the index.")
    parser.add_argument("--fused-only", action="store_true", help="Skip Jev (label validity + fused ranks only).")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    rows = load_jsonl(args.golden)
    if args.fused_only:
        passages = list_parent_passages()
        chunk_texts = _chunk_texts_by_hash()
        answerable = [row for row in rows if row.get("answerable", True)]
        fused_table = _type_recalls(answerable, config, use_jev=False)
        misses = []
        for row in answerable:
            fused = _retrieve(row, config, use_jev=False)
            fused_parents = fused.get("fused_parents") or fused.get("parents") or []
            fused_rank = _rank_in_list(fused_parents, row.get("expected") or [])
            if hit_from_rank(fused_rank, 5) == 1.0:
                continue
            label = _label_in_index(row.get("expected") or [], passages, chunk_texts)
            misses.append(
                {
                    "id": row.get("id"),
                    "type": row.get("type"),
                    "question": row.get("question"),
                    "label_valid": label["valid"],
                    "label_location": label.get("location"),
                    "fused_rank_in_30": fused_rank,
                    "jev_rank": None,
                    "reached_jev": None,
                    "jev_score": None,
                    "expected": row.get("expected"),
                }
            )
        report = {
            "answerable": len(answerable),
            "recall@5_misses": len(misses),
            "invalid_labels": [item for item in misses if not item["label_valid"]],
            "misses": misses,
            "fused": fused_table,
            "jev": None,
            "proposed_retrieval_changes": _proposals([item for item in misses if item["label_valid"]]),
            "applied": False,
            "fused_only": True,
        }
    else:
        report = analyze_misses(rows, config)

    if args.fix_invalid and report.get("invalid_labels"):
        passages = list_parent_passages()
        invalid_ids = [item["id"] for item in report["invalid_labels"]]
        updated, fixed = fix_invalid_labels(rows, invalid_ids, passages)
        write_jsonl(args.golden, updated)
        report["label_fixes"] = fixed
        print(f"Fixed {len(fixed)} invalid labels in {args.golden}", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "recall@5_misses": report["recall@5_misses"],
        "invalid_label_count": len(report.get("invalid_labels") or []),
        "fused": report.get("fused"),
        "jev": report.get("jev"),
        "proposed_retrieval_changes": report.get("proposed_retrieval_changes"),
        "misses": [
            {
                "id": item["id"],
                "type": item["type"],
                "label_valid": item["label_valid"],
                "fused_rank_in_30": item["fused_rank_in_30"],
                "jev_rank": item.get("jev_rank"),
                "reached_jev": item.get("reached_jev"),
                "jev_score": item.get("jev_score"),
            }
            for item in report.get("misses") or []
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
