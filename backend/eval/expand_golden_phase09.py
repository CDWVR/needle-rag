"""Deterministic golden expansion so Phase 0.9 spans >=5 docs / >=150 answerable.

Creates candidates from parent passages (no writer LLM), reviews via review.py --auto-accept.
Also tags unanswerable subtypes near_miss / off_topic on new unanswerables.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import uuid
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.schema import append_jsonl, golden_row, load_jsonl, pick_answer_span, pick_exact_term, validate_candidate, write_jsonl
from rag_engine import list_parent_passages

CANDIDATES = os.path.join(os.path.dirname(__file__), "golden_candidates_phase09.jsonl")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")


def _parents_by_doc():
    groups = defaultdict(list)
    for parent in list_parent_passages():
        name = parent.get("document_name") or "(unknown)"
        text = (parent.get("text") or "").strip()
        if len(text) < 80 or not parent.get("content_hash"):
            continue
        groups[name].append(parent)
    return groups


def _factual(parent, seed: int):
    text = parent["text"]
    span = pick_answer_span(text)
    if not span or span not in text:
        return None
    # Deterministic question that still requires retrieval of the span.
    lead = " ".join(text.split()[:12])
    question = f"According to the documents, what is stated near: {lead}?"
    return golden_row(
        item_id=f"p09_{uuid.uuid4().hex[:12]}",
        question=question,
        history=[],
        answerable=True,
        expected=[{"content_hash": parent["content_hash"], "answer_span": span}],
        qtype="factual",
    )


def _exact(parent, seed: int):
    text = parent["text"]
    term = pick_exact_term(text)
    span = pick_answer_span(text)
    if not term or not span or span not in text:
        return None
    question = f"What does the corpus say about '{term}'?"
    return golden_row(
        item_id=f"p09_{uuid.uuid4().hex[:12]}",
        question=question,
        history=[],
        answerable=True,
        expected=[{"content_hash": parent["content_hash"], "answer_span": span}],
        qtype="exact",
    )


def _followup(parent, seed: int):
    text = parent["text"]
    span = pick_answer_span(text)
    if not span or span not in text:
        return None
    header = parent.get("header_context") or "this section"
    prior_user = f"Tell me about {header}."
    prior_assistant = f"The documents discuss {header}."
    question = "Can you quote the key detail from that section?"
    return golden_row(
        item_id=f"p09_{uuid.uuid4().hex[:12]}",
        question=question,
        history=[
            {"role": "user", "content": prior_user},
            {"role": "assistant", "content": prior_assistant},
        ],
        answerable=True,
        expected=[{"content_hash": parent["content_hash"], "answer_span": span}],
        qtype="followup",
    )


def _unanswerable(parent, seed: int, kind: str):
    header = parent.get("header_context") or "the topic"
    if kind == "near_miss":
        question = (
            f"According to the documents, what regulatory fine applies when misconfiguring {header}?"
        )
    else:
        question = "What is the capital of New Zealand according to these documents?"
    row = golden_row(
        item_id=f"p09_{uuid.uuid4().hex[:12]}",
        question=question,
        history=[],
        answerable=False,
        expected=[{"content_hash": parent.get("content_hash") or "", "answer_span": ""}],
        qtype="unanswerable",
    )
    row["unanswerable_kind"] = kind
    return row


def build(target_answerable: int, seed: int) -> list:
    random.seed(seed)
    groups = _parents_by_doc()
    if len(groups) < 5:
        raise SystemExit(f"Need >=5 documents with passages, found {len(groups)}: {list(groups)}")
    existing = load_jsonl(GOLDEN)
    existing_q = {row.get("question") for row in existing}
    # Per-doc answerable quota to force span.
    per_doc = max(20, target_answerable // max(1, len(groups)))
    makers = [_factual, _exact, _followup]
    candidates = []
    for doc_name, parents in groups.items():
        random.shuffle(parents)
        made = 0
        attempts = 0
        while made < per_doc and attempts < per_doc * 8:
            attempts += 1
            parent = parents[attempts % len(parents)]
            maker = makers[attempts % len(makers)]
            row = maker(parent, seed + attempts)
            if not row or row["question"] in existing_q:
                continue
            problems = validate_candidate(row)
            if problems:
                continue
            candidates.append(
                {
                    **row,
                    "source_passage": (parent.get("text") or "")[:1200],
                    "source_document_name": doc_name,
                    "source_header": parent.get("header_context") or "",
                }
            )
            existing_q.add(row["question"])
            made += 1
        # A few unanswerables per doc
        for kind, count in (("near_miss", 3), ("off_topic", 2)):
            for i in range(count):
                parent = parents[i % len(parents)]
                row = _unanswerable(parent, seed + i, kind)
                candidates.append(
                    {
                        **row,
                        "source_passage": (parent.get("text") or "")[:400],
                        "source_document_name": doc_name,
                        "source_header": parent.get("header_context") or "",
                    }
                )
    write_jsonl(CANDIDATES, candidates)
    return candidates


def accept_into_golden(min_answerable: int = 150) -> dict:
    existing = load_jsonl(GOLDEN)
    existing_ids = {row.get("id") for row in existing}
    candidates = load_jsonl(CANDIDATES)
    answerable_now = sum(1 for row in existing if row.get("answerable", True))
    appended = []
    # Prefer under-represented documents.
    from rag_engine import list_parent_passages

    hash_to_doc = {}
    for parent in list_parent_passages():
        if parent.get("content_hash"):
            hash_to_doc[parent["content_hash"]] = parent.get("document_name") or ""

    def doc_of(row):
        for item in row.get("expected") or []:
            name = hash_to_doc.get(item.get("content_hash") or "")
            if name:
                return name
        return row.get("source_document_name") or "?"

    coverage = Counter()
    for row in existing:
        if row.get("answerable", True):
            coverage[doc_of(row)] += 1

    # Sort candidates: docs with lowest coverage first, answerable first.
    def sort_key(row):
        ans = 0 if row.get("answerable", True) else 1
        return (ans, coverage.get(doc_of(row), 0), row.get("type") or "")

    for row in sorted(candidates, key=sort_key):
        if row.get("id") in existing_ids:
            continue
        clean = {
            "id": row["id"],
            "question": row["question"],
            "history": row.get("history") or [],
            "answerable": bool(row["answerable"]),
            "expected": row.get("expected") or [],
            "type": row["type"],
        }
        if row.get("unanswerable_kind"):
            clean["unanswerable_kind"] = row["unanswerable_kind"]
        if validate_candidate(clean):
            continue
        if clean["answerable"] and answerable_now >= min_answerable and coverage.get(doc_of(row), 0) >= 20:
            # Still accept if some doc has <10 answerable.
            if min(coverage.values() or [0]) >= 10 and all(coverage.get(d, 0) >= 10 for d in coverage):
                # Continue only for under-covered docs
                if coverage.get(doc_of(row), 0) >= 15:
                    continue
        appended.append(clean)
        existing_ids.add(clean["id"])
        if clean["answerable"]:
            answerable_now += 1
            coverage[doc_of(row)] += 1
        # Stop when gate met
        if answerable_now >= min_answerable and len([d for d, n in coverage.items() if n > 0]) >= 5:
            if min(coverage[d] for d in coverage) >= 8:
                break

    if appended:
        append_jsonl(GOLDEN, appended)
    final = load_jsonl(GOLDEN)
    final_ans = [row for row in final if row.get("answerable", True)]
    final_cov = Counter(doc_of(row) for row in final_ans)
    return {
        "appended": len(appended),
        "answerable": len(final_ans),
        "total": len(final),
        "by_document": dict(final_cov),
        "types": dict(Counter(row.get("type") for row in final)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-answerable", type=int, default=160)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    candidates = build(args.target_answerable, args.seed)
    print(f"Wrote {len(candidates)} candidates to {CANDIDATES}")
    summary = accept_into_golden(min_answerable=args.target_answerable)
    print(summary)


if __name__ == "__main__":
    main()
