"""Pure steps from the Needle retrieval diagram.

These functions do not talk to Chroma, Jev, or Gemini. The engine calls them
so the gates can be tested without network or model downloads.
"""

import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class NeedleError(Exception):
    """Failure that should be shown to the user, not a stack trace."""


class JevNotConfigured(NeedleError):
    pass


class JevUnavailable(NeedleError):
    pass


class IncompatibleIndex(NeedleError):
    pass


def tokenize(text: str) -> List[str]:
    """Lexical tokens used before the embedding model sees the text."""
    return re.findall(r"\w+", (text or "").lower())


def index_card(header: str, parent_text: str, limit: int = 480) -> str:
    """Short searchable summary of a parent chunk."""
    lead = ""
    for line in (parent_text or "").splitlines():
        stripped = line.strip()
        if stripped:
            lead = stripped
            break
    parts = [part for part in (header.strip(), lead) if part]
    card = "\n".join(parts)
    return card[:limit]


def filter_by_similarity(
    hits: List[Dict[str, Any]],
    *,
    absolute_threshold: float,
) -> List[Dict[str, Any]]:
    """Keep vector hits at or above one cosine floor. Jev judges the rest."""
    kept = [hit for hit in hits if float(hit["similarity"]) >= absolute_threshold]
    kept.sort(key=lambda hit: float(hit["similarity"]), reverse=True)
    return kept


def contextual_passage(header: str, raw_text: str, enabled: bool) -> str:
    """Heading trail is part of the embedded string. The raw passage stays stored separately."""
    raw = raw_text or ""
    if not enabled:
        return raw
    title = (header or "").strip()
    if not title:
        return raw
    return f"{title}\n{raw}"


def normalize_query(text: str) -> str:
    return " ".join((text or "").lower().split())


def needs_condense(prior_turns: List[Dict[str, Any]]) -> bool:
    return any((turn.get("content") or "").strip() for turn in prior_turns)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    *,
    k: int,
    key: str = "chunk_id",
) -> List[Dict[str, Any]]:
    """Merge ranked lists. Rank 1 is the first item. Higher fused score wins."""
    fused: Dict[str, Dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            item_id = item[key]
            current = fused.get(item_id)
            if current is None:
                current = dict(item)
                current["rrf_score"] = 0.0
                fused[item_id] = current
            elif current.get("similarity") is None and item.get("similarity") is not None:
                current["similarity"] = item["similarity"]
            current["rrf_score"] += 1.0 / (k + rank)
    ordered = sorted(fused.values(), key=lambda item: float(item["rrf_score"]), reverse=True)
    return ordered


class ScoreCache:
    """TTL cache keyed by the caller. Used for Jev scores."""

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._values: Dict[Tuple[Any, ...], Tuple[float, float]] = {}

    def get(self, key: Tuple[Any, ...]) -> Optional[float]:
        with self._lock:
            found = self._values.get(key)
            if not found:
                return None
            score, stored_at = found
            if self._clock() - stored_at > self.ttl_seconds:
                self._values.pop(key, None)
                return None
            return score

    def put(self, key: Tuple[Any, ...], score: float) -> None:
        with self._lock:
            self._values[key] = (float(score), self._clock())


def select_parents(scored_children: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """One parent per group, ordered by the strongest Jev score inside the group."""
    best: Dict[str, Dict[str, Any]] = {}
    for child in scored_children:
        parent_id = child["parent_id"]
        current = best.get(parent_id)
        if current is None or float(child["jev_score"]) > float(current["jev_score"]):
            best[parent_id] = child
    ordered = sorted(best.values(), key=lambda item: float(item["jev_score"]), reverse=True)
    return ordered[:limit]


def verdict_passes(verdict: Optional[Dict[str, Any]]) -> bool:
    if not verdict:
        return False
    return bool(verdict.get("grounded") and verdict.get("safe") and verdict.get("relevant"))


NO_EVIDENCE_FALLBACK = (
    "I don't have enough grounded evidence in your documents to answer that. "
    "Nothing retrieved was useful enough to cite, so I am not going to guess."
)


def validation_fallback(reason: str) -> str:
    detail = reason.strip() if reason else "the draft was not clearly grounded, safe, and relevant"
    return (
        "I withheld the draft answer because it did not pass the check for grounding, safety, and relevance. "
        f"{detail}"
    )
