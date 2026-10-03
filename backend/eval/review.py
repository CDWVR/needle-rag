"""Review golden candidates and append accepted rows to golden.jsonl.

Usage:
  python backend/eval/review.py
  python backend/eval/review.py --auto-accept --min-total 60 --min-unanswerable 15 --min-followup 10 --min-exact 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.schema import (
    GOLDEN_TYPES,
    append_jsonl,
    load_jsonl,
    validate_candidate,
)

CANDIDATES = os.path.join(os.path.dirname(__file__), "golden_candidates.jsonl")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden.jsonl")


def _strip_source(row: dict) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "history": row.get("history") or [],
        "answerable": bool(row["answerable"]),
        "expected": row.get("expected") or [],
        "type": row["type"],
    }


def _counts(rows):
    tallies = {name: 0 for name in GOLDEN_TYPES}
    for row in rows:
        tallies[row.get("type")] = tallies.get(row.get("type"), 0) + 1
    tallies["total"] = len(rows)
    return tallies


def _show(row: dict, index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"Candidate {index + 1}/{total}  type={row.get('type')}  answerable={row.get('answerable')}")
    print(f"id: {row.get('id')}")
    print(f"document: {row.get('source_document_name') or '(unknown)'}")
    if row.get("source_header"):
        print(f"section: {row['source_header']}")
    print("- question -")
    print(row.get("question") or "")
    if row.get("history"):
        print("- history -")
        print(json.dumps(row["history"], indent=2, ensure_ascii=False))
    print("- expected -")
    print(json.dumps(row.get("expected") or [], indent=2, ensure_ascii=False))
    print("- source passage -")
    print((row.get("source_passage") or "")[:1000])
    problems = validate_candidate(_strip_source(row))
    if problems:
        print("VALIDATION:", "; ".join(problems))


def _edit(row: dict) -> dict:
    print("Press Enter to keep a field.")
    question = input(f"question [{row['question'][:60]}]: ").strip()
    if question:
        row["question"] = question
    span = ""
    if row.get("expected"):
        span = input(f"answer_span [{(row['expected'][0].get('answer_span') or '')[:60]}]: ").strip()
        if span:
            row["expected"][0]["answer_span"] = span
    type_name = input(f"type [{row['type']}]: ").strip()
    if type_name:
        row["type"] = type_name
    answerable = input(f"answerable [{row['answerable']}] (true/false): ").strip().lower()
    if answerable in {"true", "false"}:
        row["answerable"] = answerable == "true"
    return row


def auto_accept(candidates, existing, args) -> list:
    """Bootstrap accept path for Phase 0.5; interactive review remains the default."""
    accepted = list(existing)
    existing_ids = {item.get("id") for item in accepted}
    quotas = {
        "unanswerable": args.min_unanswerable,
        "followup": args.min_followup,
        "exact": args.min_exact,
    }

    def take(predicate):
        nonlocal accepted
        for row in candidates:
            if row.get("id") in existing_ids:
                continue
            clean = _strip_source(row)
            if validate_candidate(clean):
                continue
            if not predicate(clean, _counts(accepted)):
                continue
            accepted.append(clean)
            existing_ids.add(clean["id"])

    # Fill required type quotas first.
    for qtype, minimum in quotas.items():
        take(lambda clean, counts, qtype=qtype, minimum=minimum: clean["type"] == qtype and counts.get(qtype, 0) < minimum)
    # Pad to min_total with any valid remaining candidates.
    take(lambda clean, counts: counts["total"] < args.min_total)
    return accepted[len(existing) :]


def interactive(candidates, existing_ids) -> list:
    accepted = []
    for index, row in enumerate(candidates):
        if row.get("id") in existing_ids:
            continue
        _show(row, index, len(candidates))
        choice = input("[a]ccept / [e]dit / [r]eject / [q]uit: ").strip().lower() or "r"
        if choice.startswith("q"):
            break
        if choice.startswith("r"):
            continue
        if choice.startswith("e"):
            row = _edit(row)
        clean = _strip_source(row)
        problems = validate_candidate(clean)
        if problems:
            print("Rejected by schema:", "; ".join(problems))
            continue
        accepted.append(clean)
        print("Accepted.")
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="Review golden candidates into golden.jsonl.")
    parser.add_argument("--candidates", default=CANDIDATES)
    parser.add_argument("--golden", default=GOLDEN)
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--min-total", type=int, default=60)
    parser.add_argument("--min-unanswerable", type=int, default=15)
    parser.add_argument("--min-followup", type=int, default=10)
    parser.add_argument("--min-exact", type=int, default=10)
    args = parser.parse_args()

    candidates = load_jsonl(args.candidates)
    if not candidates:
        raise SystemExit(f"No candidates at {args.candidates}. Run build_golden.py first.")
    existing = load_jsonl(args.golden)
    existing_ids = {row.get("id") for row in existing}

    if args.auto_accept:
        accepted = auto_accept(candidates, existing, args)
    else:
        accepted = interactive(candidates, existing_ids)

    if accepted:
        append_jsonl(args.golden, accepted)
    final = load_jsonl(args.golden)
    print(json.dumps({"appended": len(accepted), "golden": _counts(final), "path": args.golden}, indent=2))


if __name__ == "__main__":
    main()
