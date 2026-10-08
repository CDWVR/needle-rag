"""Grow the workspace golden set from your own documents.

    python -m eval draft --count 30      # model drafts candidates from indexed passages (costs a little)
    python -m eval review                # accept / edit / reject them into datasets/workspace.jsonl

Drafted spans are checked to be verbatim passage text before they are written.
Nothing reaches workspace.jsonl without a human accepting it.
"""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from typing import Any, Dict, List, Optional

from eval.schema import append_jsonl, load_jsonl, span_in_text, validate_dataset, validate_row, write_jsonl

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(EVAL_DIR, "datasets", "workspace.jsonl")
CANDIDATES = os.path.join(EVAL_DIR, "datasets", "workspace.candidates.jsonl")

_KINDS = {
    "factual": "a factual question a colleague would ask, answered by one sentence of the passage",
    "exact": "a question that hinges on an exact name, code, number, or date in the passage",
    "unanswerable": (
        "a realistic question on the same topic that the passage does NOT answer (a near miss); "
        "answer_span must be empty"
    ),
}


def _ask(prompt: str) -> Dict[str, Any]:
    from llm import complete

    raw = complete(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
        cost_purpose="golden_build",
    )
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def draft(count: int, seed: int = 7) -> int:
    from catalog import list_parent_passages
    from llm import cost_ledger_snapshot

    passages = [p for p in list_parent_passages() if len((p.get("text") or "").strip()) >= 200]
    if not passages:
        print("No indexed passages to draft from.")
        return 0
    existing = {" ".join(row["question"].lower().split()) for row in load_jsonl(WORKSPACE) + load_jsonl(CANDIDATES)}
    rng = random.Random(seed)
    kinds = ["factual"] * 5 + ["exact"] * 3 + ["unanswerable"] * 2
    written = 0
    attempts = 0
    while written < count and attempts < count * 3:
        attempts += 1
        passage = rng.choice(passages)
        kind = rng.choice(kinds)
        parsed = _ask(
            "Write one evaluation question for a document assistant. Return JSON only: "
            '{"question": "...", "answer_span": "...", "key_facts": ["..."]}. '
            f"Make it {_KINDS[kind]}. answer_span must be copied character for character from the passage "
            "(a short phrase or sentence). key_facts are 1-3 short strings a correct answer must contain. "
            "Do not mention 'the passage' or 'the document' in the question.\n\n"
            f"DOCUMENT: {passage['document_name']}\nPASSAGE:\n{passage['text'][:1800]}"
        )
        question = " ".join(str(parsed.get("question") or "").split())
        span = str(parsed.get("answer_span") or "").strip()
        if not question or " ".join(question.lower().split()) in existing:
            continue
        answerable = kind != "unanswerable"
        if answerable and (not span or not span_in_text(passage["text"], span)):
            continue  # the model paraphrased instead of quoting; drop it rather than store a wrong anchor
        row = {
            "id": f"ws-{uuid.uuid4().hex[:10]}",
            "type": kind,
            "answerable": answerable,
            "question": question,
            "expected": [{"document": passage["document_name"], "content_hash": passage.get("content_hash") or "",
                          "answer_span": span}] if answerable else [],
            "key_facts": [str(f) for f in parsed.get("key_facts") or [] if str(f).strip()][:3] if answerable else [],
            "tags": ["drafted"],
            "_source": {"document": passage["document_name"], "page": passage.get("page_number"),
                        "excerpt": passage["text"][:600]},
        }
        if not row["expected"]:
            row.pop("key_facts")
        elif not row["expected"][0]["content_hash"]:
            row["expected"][0].pop("content_hash")
        if validate_row({k: v for k, v in row.items() if not k.startswith("_")}):
            continue
        append_jsonl(CANDIDATES, [row])
        existing.add(" ".join(question.lower().split()))
        written += 1
    print(f"Drafted {written} candidates into {os.path.relpath(CANDIDATES)} (spend shown under golden_build).")
    print(f"Spend: {cost_ledger_snapshot()['golden_build']:.4f} USD")
    return written


def _show(row: Dict[str, Any], index: int, total: int) -> None:
    source = row.get("_source") or {}
    print("\n" + "=" * 72)
    print(f"[{index + 1}/{total}] {row['type']} · {source.get('document')} p.{source.get('page')}")
    print(f"Q: {row['question']}")
    for item in row.get("expected") or []:
        print(f"Span: {item.get('answer_span')!r}")
    if row.get("key_facts"):
        print(f"Key facts: {row['key_facts']}")
    print(f"Excerpt: {(source.get('excerpt') or '')[:400]}")


def review() -> int:
    candidates = load_jsonl(CANDIDATES)
    if not candidates:
        print(f"No candidates in {os.path.relpath(CANDIDATES)}. Run `python -m eval draft` first.")
        return 0
    accepted: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for index, row in enumerate(candidates):
        _show(row, index, len(candidates))
        choice = (input("[a]ccept / [e]dit question / [r]eject / [s]kip / [q]uit: ").strip().lower() or "s")[0]
        if choice == "q":
            remaining.extend(candidates[index:])
            break
        if choice == "s":
            remaining.append(row)
            continue
        if choice == "r":
            continue
        if choice == "e":
            edited = input("New question (blank keeps it): ").strip()
            if edited:
                row["question"] = edited
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        problems = validate_row(clean)
        if problems:
            print("Not accepted: " + "; ".join(problems))
            remaining.append(row)
            continue
        accepted.append(clean)
    dataset = load_jsonl(WORKSPACE) + accepted
    problems = validate_dataset(dataset)
    if problems:
        print("Accepted rows would make the dataset invalid; nothing written:\n  " + "\n  ".join(problems[:10]))
        return 0
    write_jsonl(WORKSPACE, dataset)
    write_jsonl(CANDIDATES, remaining)
    print(f"\nAdded {len(accepted)} rows to {os.path.relpath(WORKSPACE)}; {len(remaining)} candidates left.")
    return len(accepted)


def default_count(value: Optional[int]) -> int:
    return max(1, int(value or 20))
