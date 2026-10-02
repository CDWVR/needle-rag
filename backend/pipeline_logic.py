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


class CircuitBreaker:
    """Opens after repeated failures and closes again once the cooldown has passed."""

    def __init__(self, failure_limit: int, reset_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.failure_limit = failure_limit
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0

    def closed(self) -> bool:
        with self._lock:
            if self._open_until and self._clock() >= self._open_until:
                self._failures = 0
                self._open_until = 0.0
            return self._open_until == 0.0

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_limit:
                self._open_until = self._clock() + self.reset_seconds

    def status(self) -> str:
        return "closed" if self.closed() else "open"


def confidence_bucket(score: float, *, medium: float, high: float) -> str:
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def rank_summary(scored: List[Dict[str, Any]], *, keep_threshold: float) -> Dict[str, Any]:
    ordered = sorted(scored, key=lambda item: float(item.get("score") or 0), reverse=True)
    top = float(ordered[0]["score"]) if ordered else 0.0
    kept = [item for item in ordered if float(item.get("score") or 0) >= keep_threshold]
    return {"ordered": ordered, "top_score": top, "kept": kept, "kept_count": len(kept)}


def fused_fallback_scores(count: int) -> List[float]:
    """Rank-only scores used when neither Jev nor a local cross-encoder is available."""
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [1.0 - (index / (count - 1)) * 0.5 for index in range(count)]


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


_INJECTION = (
    re.compile(r"ignore (all |any |the )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (the )?(system|previous|prior)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
)


def passage_looks_like_instructions(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _INJECTION)


def split_sentences(answer: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (answer or "").strip())
    return [part.strip() for part in parts if part.strip()]


def kept_sentences(answer: str, unsupported_indexes: List[int], *, minimum_chars: int) -> Dict[str, Any]:
    sentences = split_sentences(answer)
    blocked = set(unsupported_indexes)
    kept = [sentence for index, sentence in enumerate(sentences, start=1) if index not in blocked]
    text = " ".join(kept).strip()
    return {
        "text": text,
        "partially_supported": bool(sentences) and len(kept) < len(sentences) and bool(kept),
        "enough": len(text) >= minimum_chars,
    }


def deterministic_violations(answer: str, contexts: List[Dict[str, Any]], *, min_quote_chars: int) -> List[str]:
    """Fail closed when citations, quotations, or numbers are not in the cited text."""
    problems = []
    cited = [int(number) for number in re.findall(r"\[(\d+)\]", answer or "")]
    limit = len(contexts)
    for number in cited:
        if number < 1 or number > limit:
            problems.append(f"Citation [{number}] does not match a retrieved passage.")
    corpus = "\n".join(str(context.get("text") or "") for context in contexts)
    for quote in re.findall(r"[\"“](.{8,}?)[\"”]", answer or "", flags=re.S):
        snippet = quote.strip()
        if len(snippet) >= min_quote_chars and snippet not in corpus:
            problems.append("A quotation in the answer is not in the cited passages.")
            break
    for number in re.findall(r"\d[\d,]*(?:\.\d+)?", answer or ""):
        if re.fullmatch(r"\d", number):
            continue
        plain = number.replace(",", "")
        if number not in corpus and plain not in corpus:
            problems.append(f"The number {number} is not in the cited passages.")
            break
    return problems


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
