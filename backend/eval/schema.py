"""Golden-set schema and evidence matching.

A golden row never stores Chroma or SQLite ids, which change on every re-index.
Evidence is anchored by the source document (its file name, optionally also the
SHA-256 of the uploaded bytes) plus a verbatim `answer_span` that must appear in
a retrieved parent passage.

    {
      "id": "rnn-001",
      "question": "What problem do LSTMs address?",
      "history": [],                       # prior turns for follow-ups: [{"role", "content"}]
      "type": "factual",                   # factual | exact | multihop | followup | unanswerable
      "answerable": true,
      "expected": [{"document": "rnn_basics.txt", "answer_span": "vanishing gradient"}],
      "key_facts": ["vanishing gradient"], # optional: each must appear in a correct answer ("a|b" = either)
      "forbidden_phrases": ["..."],        # optional: must never appear in a released answer
      "standalone_question": "...",        # required with history: the offline tier searches with it
      "reference_answer": "...",           # optional: used by the LLM judge
      "tags": ["smoke"]                    # optional
    }

Expected items may also carry "page" (1-based) to check page-level citation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

GOLDEN_TYPES = ("factual", "exact", "multihop", "followup", "unanswerable")
_FORBIDDEN_KEYS = ("document_id", "chunk_id", "parent_id", "collection_name")


def _normalize(text: str) -> str:
    # Local copy of pipeline_logic.normalize_match_text so CI can import this without the engine.
    cleaned = (text or "").replace(" ", " ").replace(" ", " ").replace(" ", " ")
    for left, right in (("“", '"'), ("”", '"'), ("„", '"'), ("‘", "'"), ("’", "'"), ("–", "-"), ("—", "-"),
                        ("−", "-"), ("­", ""), ("…", "...")):
        cleaned = cleaned.replace(left, right)
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def span_in_text(text: str, span: str) -> bool:
    if not span:
        return True
    return span in (text or "") or _normalize(span) in _normalize(text)


# "What is the title of the document?" only makes sense next to the passage it was written from.
_DECONTEXTUALISED = re.compile(
    r"\b(?:the|this|that|above|given|provided)\s+(?:passage|excerpt|text|snippet|section|chunk|document|article)\b"
    r"|\bmentioned in the\b|\baccording to the (?:passage|text|excerpt)\b"
    # Template questions from the old generator: "What does the corpus say about 'Ideally'?"
    r"|^what does the corpus say about\b|\bwhat is mentioned about\b|\bthe heading that contains the token\b",
    re.I,
)


def is_decontextualised(question: str) -> bool:
    return bool(_DECONTEXTUALISED.search(question or ""))


def validate_row(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not str(row.get("id") or "").strip():
        errors.append("missing id")
    if not str(row.get("question") or "").strip():
        errors.append("missing question")
    elif is_decontextualised(row.get("question") or ""):
        errors.append("question points at 'the passage/document'; rewrite it so it stands alone")
    qtype = row.get("type")
    if qtype not in GOLDEN_TYPES:
        errors.append(f"bad type {qtype!r}")
    if not isinstance(row.get("answerable"), bool):
        errors.append("answerable must be true or false")
    history = row.get("history", [])
    if not isinstance(history, list) or any(
        not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"} or not str(turn.get("content") or "").strip()
        for turn in history
    ):
        errors.append("history must be a list of {role: user|assistant, content}")
    if qtype == "followup" and not history:
        errors.append("followup rows need history")
    expected = row.get("expected", [])
    if not isinstance(expected, list):
        errors.append("expected must be a list")
        expected = []
    for item in expected:
        if not isinstance(item, dict):
            errors.append("expected item must be an object")
            continue
        if any(key in item for key in _FORBIDDEN_KEYS):
            errors.append("expected must not reference database or Chroma ids")
        if not (item.get("document") or item.get("content_hash")):
            errors.append("expected item needs a document name or content_hash")
        if "page" in item and not (isinstance(item["page"], int) and item["page"] >= 1):
            errors.append("expected page must be a positive integer")
    if row.get("answerable") is True:
        if not expected:
            errors.append("answerable rows need at least one expected passage")
        if any(not str(item.get("answer_span") or "").strip() for item in expected if isinstance(item, dict)):
            errors.append("answerable expected items need an answer_span")
        if qtype == "unanswerable":
            errors.append("type unanswerable requires answerable=false")
    if row.get("answerable") is False and qtype != "unanswerable":
        errors.append("answerable=false requires type unanswerable")
    for field in ("key_facts", "tags", "forbidden_phrases"):
        value = row.get(field)
        if value is not None and (not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value)):
            errors.append(f"{field} must be a list of non-empty strings")
    standalone = row.get("standalone_question")
    if standalone is not None and not (isinstance(standalone, str) and standalone.strip()):
        errors.append("standalone_question must be a non-empty string")
    if history and not standalone:
        errors.append("rows with history need standalone_question (used by the offline tier)")
    return errors


def validate_dataset(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Row errors plus duplicate ids and duplicate (question, history) pairs, which inflate metrics."""
    problems: List[str] = []
    seen_ids: Dict[str, int] = {}
    seen_questions: Dict[str, str] = {}
    for line, row in enumerate(rows, start=1):
        for error in validate_row(row):
            problems.append(f"row {line} ({row.get('id')}): {error}")
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            problems.append(f"row {line}: duplicate id {row_id!r} (first on row {seen_ids[row_id]})")
        seen_ids.setdefault(row_id, line)
        key = json.dumps([_normalize(row.get("question") or ""), row.get("history") or []], sort_keys=True)
        if key in seen_questions:
            problems.append(f"row {line} ({row_id}): duplicate question of {seen_questions[key]}")
        seen_questions.setdefault(key, row_id)
    return problems


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    materialised = list(rows)
    with open(path, "w", encoding="utf-8") as handle:
        for row in materialised:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(materialised)


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _same_document(parent: Dict[str, Any], item: Dict[str, Any]) -> bool:
    name = str(item.get("document") or "").strip()
    if name and name == str(parent.get("document_name") or "").strip():
        return True
    content_hash = str(item.get("content_hash") or "").strip()
    return bool(content_hash) and content_hash == str(parent.get("content_hash") or "").strip()


def parent_matches(parent: Dict[str, Any], item: Dict[str, Any]) -> bool:
    return _same_document(parent, item) and span_in_text(parent.get("text") or "", str(item.get("answer_span") or ""))


def first_match_rank(parents: Sequence[Dict[str, Any]], expected: Sequence[Dict[str, Any]]) -> Optional[int]:
    """1-based rank of the first parent that contains any expected passage."""
    for rank, parent in enumerate(parents, start=1):
        if any(parent_matches(parent, item) for item in expected):
            return rank
    return None


def documents_covered(row: Dict[str, Any], available: Dict[str, set]) -> bool:
    """True when every expected document is present in the index being evaluated.

    `available` is {"names": {...}, "hashes": {...}}.
    """
    for item in row.get("expected") or []:
        name = str(item.get("document") or "").strip()
        content_hash = str(item.get("content_hash") or "").strip()
        if not ((name and name in available.get("names", set())) or (content_hash and content_hash in available.get("hashes", set()))):
            return False
    return True
