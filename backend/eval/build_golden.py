"""Sample parents and ask the writer (temperature 0) for golden candidates.

Usage from the repo root:
  python backend/eval/build_golden.py
  python backend/eval/build_golden.py --target 90

Writes backend/eval/golden_candidates.jsonl only. Never touches golden.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import uuid
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.schema import (
    GOLDEN_TYPES,
    append_jsonl,
    golden_row,
    pick_answer_span,
    pick_exact_term,
    write_jsonl,
)
from rag_engine import OPENROUTER_MODEL, _guarded_writer, list_parent_passages


OUT_PATH = os.path.join(os.path.dirname(__file__), "golden_candidates.jsonl")


def _ask_json(prompt: str) -> dict:
    raw = _guarded_writer(
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
        model=OPENROUTER_MODEL,
    )
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _unique_parents(parents):
    seen = set()
    unique = []
    for parent in parents:
        key = (parent.get("content_hash"), (parent.get("text") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(parent)
    return unique


def _candidate_hard(parent, other_parent, style: str, seed: int) -> Optional[dict]:
    """Harder cases mapped onto the migration-safe type enum."""
    text = (parent.get("text") or "").strip()
    content_hash = parent.get("content_hash") or ""
    if len(text) < 40 or not content_hash:
        return None
    span = pick_answer_span(text)
    if style == "paraphrase":
        parsed = _ask_json(
            "Return JSON only with keys question and answer_span. "
            "Write a paraphrased question with low lexical overlap to the passage "
            "(avoid copying rare nouns from the passage into the question). "
            "answer_span must be a verbatim substring of the passage.\n\n"
            f"PASSAGE:\n{text[:1400]}"
        )
        qtype = "factual"
    elif style == "table_list":
        parsed = _ask_json(
            "Return JSON only with keys question and answer_span. "
            "Ask about an item that appears in a list, bullet, numbered step, or table-like row "
            "in the passage. answer_span must be a verbatim substring of the passage.\n\n"
            f"PASSAGE:\n{text[:1400]}"
        )
        qtype = "exact"
    elif style == "multidoc":
        other_text = (other_parent.get("text") or "").strip() if other_parent else ""
        other_hash = (other_parent or {}).get("content_hash") or content_hash
        parsed = _ask_json(
            "Return JSON only with keys question, answer_span, and secondary_span. "
            "Write one question that needs facts from BOTH passages below. "
            "answer_span must be verbatim from PASSAGE A; secondary_span verbatim from PASSAGE B.\n\n"
            f"PASSAGE A:\n{text[:900]}\n\nPASSAGE B:\n{other_text[:900]}"
        )
        qtype = "multihop"
        question = (parsed.get("question") or "").strip()
        span_a = (parsed.get("answer_span") or span).strip()
        span_b = (parsed.get("secondary_span") or "").strip()
        if not question or span_a not in text:
            return None
        expected = [{"content_hash": content_hash, "answer_span": span_a}]
        if span_b and other_text and span_b in other_text:
            expected.append({"content_hash": other_hash, "answer_span": span_b})
        return golden_row(
            item_id=f"cand_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[],
            answerable=True,
            expected=expected,
            qtype=qtype,
        )
    elif style == "near_dup":
        parsed = _ask_json(
            "Return JSON only with keys question and answer_span. "
            "Write a question that could match near-duplicate copies of the same document. "
            "answer_span must be a verbatim substring of the passage.\n\n"
            f"PASSAGE:\n{text[:1400]}"
        )
        qtype = "factual"
    elif style == "hard_followup":
        parsed = _ask_json(
            "Return JSON only with keys prior_user, prior_assistant, question, answer_span. "
            "Create a follow-up that depends on the prior turn and cannot be answered from the "
            "follow-up alone. answer_span must be a verbatim substring of the passage.\n\n"
            f"PASSAGE:\n{text[:1400]}"
        )
        question = (parsed.get("question") or "").strip()
        span = (parsed.get("answer_span") or span).strip()
        if not question or span not in text:
            return None
        return golden_row(
            item_id=f"cand_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[
                {"role": "user", "content": (parsed.get("prior_user") or "Summarize the section.").strip()},
                {"role": "assistant", "content": (parsed.get("prior_assistant") or "Here is a short overview.").strip()},
            ],
            answerable=True,
            expected=[{"content_hash": content_hash, "answer_span": span}],
            qtype="followup",
        )
    else:
        return None
    question = (parsed.get("question") or "").strip()
    span = (parsed.get("answer_span") or span).strip()
    if not question or span not in text:
        return None
    return golden_row(
        item_id=f"cand_{uuid.uuid4().hex[:12]}",
        question=question,
        history=[],
        answerable=True,
        expected=[{"content_hash": content_hash, "answer_span": span}],
        qtype=qtype,
    )


def _candidate_from_parent(parent, qtype: str, seed: int) -> Optional[dict]:
    text = (parent.get("text") or "").strip()
    if len(text) < 40 and qtype != "unanswerable":
        return None
    content_hash = parent.get("content_hash") or ""
    if not content_hash:
        return None
    span = pick_answer_span(text)
    term = pick_exact_term(text)
    header = parent.get("header_context") or parent.get("document_name") or "the passage"

    if qtype == "exact":
        if not term:
            return None
        question = f"What does the document say about {term}?"
        parsed = _ask_json(
            "Return JSON only with keys question and answer_span. "
            "Write one exact-term lookup question that requires the token "
            f"'{term}' to be found via keyword search. answer_span must be a "
            "verbatim substring of the passage.\n\nPASSAGE:\n"
            f"{text[:1200]}"
        )
        question = (parsed.get("question") or question).strip()
        span = (parsed.get("answer_span") or span).strip()
        if span not in text:
            span = pick_answer_span(text)
        return golden_row(
            item_id=f"cand_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[],
            answerable=True,
            expected=[{"content_hash": content_hash, "answer_span": span}],
            qtype="exact",
        )

    if qtype == "factual":
        parsed = _ask_json(
            "Return JSON only with keys question and answer_span. "
            "Write one factual lookup question answerable only from this passage. "
            "answer_span must be a verbatim substring of the passage (20-160 chars).\n\n"
            f"SECTION: {header}\nPASSAGE:\n{text[:1400]}"
        )
        question = (parsed.get("question") or "").strip()
        span = (parsed.get("answer_span") or span).strip()
        if not question or span not in text:
            return None
        return golden_row(
            item_id=f"cand_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[],
            answerable=True,
            expected=[{"content_hash": content_hash, "answer_span": span}],
            qtype="factual",
        )

    if qtype == "multihop":
        parsed = _ask_json(
            "Return JSON only with keys question and answer_span. "
            "Write one multi-sentence question that needs more than a single keyword hit "
            "but is still answerable from this passage. "
            "answer_span must be a verbatim substring of the passage.\n\n"
            f"PASSAGE:\n{text[:1400]}"
        )
        question = (parsed.get("question") or "").strip()
        span = (parsed.get("answer_span") or span).strip()
        if not question or span not in text:
            return None
        return golden_row(
            item_id=f"cand_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[],
            answerable=True,
            expected=[{"content_hash": content_hash, "answer_span": span}],
            qtype="multihop",
        )

    if qtype == "followup":
        parsed = _ask_json(
            "Return JSON only with keys prior_user, prior_assistant, question, answer_span. "
            "Create a short two-turn history and a follow-up question whose answer is in the passage. "
            "answer_span must be a verbatim substring of the passage.\n\n"
            f"PASSAGE:\n{text[:1400]}"
        )
        question = (parsed.get("question") or "").strip()
        span = (parsed.get("answer_span") or span).strip()
        prior_user = (parsed.get("prior_user") or f"Tell me about {header}.").strip()
        prior_assistant = (parsed.get("prior_assistant") or "Here is a short overview from the documents.").strip()
        if not question or span not in text:
            return None
        return golden_row(
            item_id=f"cand_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[
                {"role": "user", "content": prior_user},
                {"role": "assistant", "content": prior_assistant},
            ],
            answerable=True,
            expected=[{"content_hash": content_hash, "answer_span": span}],
            qtype="followup",
        )

    if qtype == "unanswerable":
        # Mix: topic-present but answer absent, and fully off-topic.
        if seed % 2 == 0:
            question = (
                f"According to the documents, what is the regulatory fine amount for "
                f"misconfiguring {term or header} in production?"
            )
            parsed = _ask_json(
                "Return JSON only with key question. Write one question about a detail "
                "that sounds related to the passage topic but is NOT answered in the passage "
                "(invent a specific missing metric, date, or owner).\n\n"
                f"PASSAGE:\n{text[:900]}"
            )
            question = (parsed.get("question") or question).strip()
        else:
            question = "What is the capital of New Zealand according to these documents?"
        return golden_row(
            item_id=f"cand_{uuid.uuid4().hex[:12]}",
            question=question,
            history=[],
            answerable=False,
            expected=[{"content_hash": content_hash, "answer_span": ""}],
            qtype="unanswerable",
        )
    return None


def build(target: int, seed: int) -> list:
    random.seed(seed)
    parents = _unique_parents(list_parent_passages())
    if not parents:
        raise SystemExit("No parent passages found in the active index.")
    # Prefer longer parents for answerable types.
    ranked = sorted(parents, key=lambda item: len(item.get("text") or ""), reverse=True)
    hard_budget = max(20, target // 3)
    quotas = {
        "factual": max(12, (target - hard_budget) // 5),
        "exact": max(12, (target - hard_budget) // 6),
        "multihop": max(10, (target - hard_budget) // 7),
        "followup": max(12, (target - hard_budget) // 6),
        "unanswerable": max(18, (target - hard_budget) // 4),
    }
    # Normalize to at least the target.
    while sum(quotas.values()) + hard_budget < target:
        quotas["factual"] += 1

    candidates = []
    for qtype in GOLDEN_TYPES:
        need = quotas[qtype]
        attempts = 0
        while need > 0 and attempts < need * 4:
            attempts += 1
            parent = ranked[(attempts + seed) % len(ranked)]
            try:
                row = _candidate_from_parent(parent, qtype, seed + attempts)
            except Exception as exc:
                print(f"skip {qtype}: {exc}", file=sys.stderr)
                continue
            if not row:
                continue
            candidates.append({
                **row,
                "source_passage": (parent.get("text") or "")[:1200],
                "source_document_name": parent.get("document_name") or "",
                "source_header": parent.get("header_context") or "",
            })
            need -= 1
            print(f"[{len(candidates)}] {qtype}: {row['question'][:80]}")

    hard_styles = ["paraphrase", "table_list", "multidoc", "near_dup", "hard_followup"]
    attempts = 0
    while hard_budget > 0 and attempts < hard_budget * 5:
        attempts += 1
        parent = ranked[attempts % len(ranked)]
        other = ranked[(attempts * 3) % len(ranked)]
        style = hard_styles[attempts % len(hard_styles)]
        try:
            row = _candidate_hard(parent, other, style, seed + attempts)
        except Exception as exc:
            print(f"skip hard/{style}: {exc}", file=sys.stderr)
            continue
        if not row:
            continue
        candidates.append({
            **row,
            "source_passage": (parent.get("text") or "")[:1200],
            "source_document_name": parent.get("document_name") or "",
            "source_header": parent.get("header_context") or "",
            "hard_style": style,
        })
        hard_budget -= 1
        print(f"[{len(candidates)}] hard/{style}/{row['type']}: {row['question'][:80]}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build golden candidates for human review.")
    parser.add_argument("--target", type=int, default=90, help="Approximate candidate count")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--append", action="store_true", help="Append to existing candidates file")
    args = parser.parse_args()
    rows = build(args.target, args.seed)
    count = append_jsonl(args.out, rows) if args.append else write_jsonl(args.out, rows)
    print(json.dumps({"wrote": count, "path": args.out, "by_type": _counts(rows), "append": args.append}, indent=2))


def _counts(rows):
    tallies = {name: 0 for name in GOLDEN_TYPES}
    for row in rows:
        tallies[row["type"]] = tallies.get(row["type"], 0) + 1
    return tallies


if __name__ == "__main__":
    main()
