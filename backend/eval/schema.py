"""Migration-safe golden schema helpers.

Golden rows never store Chroma or SQLite ids. Matching is by document
content_hash plus a verbatim answer_span found inside a retrieved parent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

GOLDEN_TYPES = ("factual", "exact", "multihop", "followup", "unanswerable")


def content_hash_from_texts(texts: Sequence[str]) -> str:
    blob = "\n".join(sorted((text or "").strip() for text in texts if (text or "").strip()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def pick_answer_span(text: str, *, min_chars: int = 24, max_chars: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    # Prefer a sentence-like window that still fits.
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    for part in parts:
        part = part.strip()
        if min_chars <= len(part) <= max_chars:
            return part
    return cleaned[:max_chars].rsplit(" ", 1)[0].strip() or cleaned[:max_chars]


def pick_exact_term(text: str) -> str:
    cleaned = text or ""
    # CamelCase / code-like / Title Case tokens first, then longer words.
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_\-]{2,}\b", cleaned)
    if not candidates:
        candidates = [w for w in re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{4,}\b", cleaned) if w.lower() not in {
            "about", "these", "those", "where", "which", "their", "there", "using", "based"
        }]
    if not candidates:
        return ""
    return max(candidates, key=len)


def validate_candidate(row: Dict[str, Any]) -> List[str]:
    errors = []
    if not row.get("id"):
        errors.append("missing id")
    if not (row.get("question") or "").strip():
        errors.append("missing question")
    qtype = row.get("type")
    if qtype not in GOLDEN_TYPES:
        errors.append(f"bad type {qtype!r}")
    if "answerable" not in row:
        errors.append("missing answerable")
    history = row.get("history")
    if history is None:
        errors.append("missing history")
    elif not isinstance(history, list):
        errors.append("history must be a list")
    expected = row.get("expected")
    if expected is None:
        errors.append("missing expected")
    elif not isinstance(expected, list):
        errors.append("expected must be a list")
    else:
        for item in expected:
            if not isinstance(item, dict):
                errors.append("expected item must be an object")
                continue
            if any(key in item for key in ("document_id", "chunk_id", "parent_id", "collection_name")):
                errors.append("expected must not reference DB or Chroma ids")
            if not item.get("content_hash"):
                errors.append("expected item missing content_hash")
            if row.get("answerable") and not (item.get("answer_span") or "").strip():
                errors.append("answerable item missing answer_span")
    if row.get("answerable") is True and qtype == "unanswerable":
        errors.append("unanswerable type must set answerable=false")
    if row.get("answerable") is False and qtype != "unanswerable":
        errors.append("answerable=false requires type=unanswerable")
    return errors


def golden_row(
    *,
    item_id: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    answerable: bool,
    expected: List[Dict[str, str]],
    qtype: str,
) -> Dict[str, Any]:
    row = {
        "id": item_id,
        "question": question.strip(),
        "history": history or [],
        "answerable": bool(answerable),
        "expected": expected,
        "type": qtype,
    }
    problems = validate_candidate(row)
    if problems:
        raise ValueError("; ".join(problems))
    return row


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    materialised = list(rows)
    with open(path, "w", encoding="utf-8") as handle:
        for row in materialised:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(materialised)


def expected_hashes(row: Dict[str, Any]) -> List[str]:
    return [item["content_hash"] for item in row.get("expected") or [] if item.get("content_hash")]


def parent_matches_expected(parent: Dict[str, Any], expected: Sequence[Dict[str, str]]) -> bool:
    from pipeline_logic import corpus_contains_span

    text = parent.get("text") or ""
    content_hash = parent.get("content_hash") or ""
    for item in expected:
        if content_hash and content_hash == item.get("content_hash"):
            span = item.get("answer_span") or ""
            if not span or corpus_contains_span(text, span):
                return True
    return False


def first_match_rank(retrieved_parents: Sequence[Dict[str, Any]], expected: Sequence[Dict[str, str]]) -> Optional[int]:
    for index, parent in enumerate(retrieved_parents, start=1):
        if parent_matches_expected(parent, expected):
            return index
    return None
