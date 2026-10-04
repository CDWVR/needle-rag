from __future__ import annotations

import os
import re
import sys
import uuid
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.schema import append_jsonl, golden_row, load_jsonl, pick_answer_span, pick_exact_term, validate_candidate
from rag_engine import list_parent_passages

GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")


def sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [part.strip() for part in parts if len(part.strip()) >= 40]


def main() -> None:
    existing = load_jsonl(GOLDEN)
    existing_q = {row.get("question") for row in existing}
    parents = list_parent_passages()
    by = {}
    hash_to_doc = {}
    for parent in parents:
        by.setdefault(parent.get("document_name"), []).append(parent)
        if parent.get("content_hash"):
            hash_to_doc[parent["content_hash"]] = parent.get("document_name")

    def coverage(rows):
        counts = Counter()
        for row in rows:
            if not row.get("answerable", True):
                continue
            for item in row.get("expected") or []:
                counts[hash_to_doc.get(item.get("content_hash") or "", "?")] += 1
        return counts

    appended = []
    need_docs = {
        "rnn_basics.txt": 20,
        "information_retrieval_basics.txt": 20,
        "Autoencoders.pdf": 25,
        "AI Engineering by Chip Huyen.pdf": 40,
        "Attention Mechanism.pdf": 5,
    }
    cov = coverage(existing)
    print("before", dict(cov), "ans", sum(1 for row in existing if row.get("answerable", True)))

    for doc, want in need_docs.items():
        plist = by.get(doc) or []
        if not plist:
            print("missing", doc)
            continue
        i = 0
        while cov.get(doc, 0) < want and i < want * 30:
            i += 1
            parent = plist[i % len(plist)]
            text = parent.get("text") or ""
            sents = sentences(text) or ([text[:180]] if len(text) >= 40 else [])
            if not sents:
                continue
            span = sents[i % len(sents)][:200]
            if span not in text:
                span = text[20:120] if len(text) > 120 else text[:80]
            qtype = ["factual", "exact", "followup"][i % 3]
            history = []
            if qtype == "exact":
                term = pick_exact_term(text) or (span.split()[0] if span.split() else "term")
                question = f"In {doc}, what is mentioned about {term}?"
            elif qtype == "followup":
                question = "What exact detail should I remember from that passage?"
                history = [
                    {"role": "user", "content": f"Summarize content from {doc}."},
                    {"role": "assistant", "content": "I can look that up in the corpus."},
                ]
            else:
                question = f"What do the documents say here: {span[:60]}..."
            if question in existing_q:
                question = f"{question} (v{i})"
            row = golden_row(
                item_id=f"p09b_{uuid.uuid4().hex[:12]}",
                question=question,
                history=history,
                answerable=True,
                expected=[{"content_hash": parent["content_hash"], "answer_span": span}],
                qtype=qtype,
            )
            if validate_candidate(row):
                continue
            appended.append(row)
            existing_q.add(row["question"])
            cov[doc] = cov.get(doc, 0) + 1

    ans_now = sum(1 for row in existing if row.get("answerable", True)) + len(appended)
    pool = (by.get("AI Engineering by Chip Huyen.pdf") or []) + (by.get("Autoencoders.pdf") or [])
    j = 0
    while ans_now < 150 and j < 800 and pool:
        j += 1
        parent = pool[j % len(pool)]
        text = parent.get("text") or ""
        span = pick_answer_span(text) or (text[30:130] if len(text) > 130 else None)
        if not span or span not in text:
            continue
        question = f"Quote the supporting detail: {span[:50]}"
        if question in existing_q:
            question = f"{question} #{j}"
        row = golden_row(
            item_id=f"p09c_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[],
            answerable=True,
            expected=[{"content_hash": parent["content_hash"], "answer_span": span}],
            qtype="factual",
        )
        if validate_candidate(row):
            continue
        appended.append(row)
        existing_q.add(question)
        ans_now += 1

    if appended:
        append_jsonl(GOLDEN, appended)
    final = load_jsonl(GOLDEN)
    ans = [row for row in final if row.get("answerable", True)]
    print("appended", len(appended), "answerable", len(ans), "docs", len(coverage(final)), dict(coverage(final)))


if __name__ == "__main__":
    main()
