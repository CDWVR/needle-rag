"""Pure steps from the Needle retrieval diagram.

These functions do not talk to Chroma, Jev, or Gemini. The engine calls them
so the gates can be tested without network or model downloads.
"""

import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # Any used by violation details


class NeedleError(Exception):
    """Failure that should be shown to the user, not a stack trace."""


class JevNotConfigured(NeedleError):
    pass


class JevUnavailable(NeedleError):
    pass


class IncompatibleIndex(NeedleError):
    pass


class InfraError(NeedleError):
    """Upstream infra failure (OpenRouter 403/429/5xx/timeout). Not an abstention."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, kind: str = "infra_error"):
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind


def classify_http_infra_error(status_code: Optional[int], *, timeout: bool = False) -> Optional[str]:
    if timeout:
        return "timeout"
    if status_code is None:
        return None
    if status_code == 403:
        return "key_limit_or_forbidden"
    if status_code == 429:
        return "rate_limit"
    if status_code >= 500:
        return "upstream_5xx"
    return None


def infra_error_share(infra_count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(infra_count) / float(total)


def infra_errors_exceed_share(infra_count: int, total: int, *, max_share: float = 0.02) -> bool:
    return infra_error_share(infra_count, total) > float(max_share) + 1e-12


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


RETRIEVAL_RERANK_MODES = ("jev_filter", "fused_only", "jev_soft")


def soft_rrf_ranks(
    fused_ranks: Dict[str, int],
    jev_ranks: Dict[str, int],
    *,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """RRF blend of fused and Jev ranks. Higher score is better."""
    ids = set(fused_ranks) | set(jev_ranks)
    scored = []
    for item_id in ids:
        fused_rank = fused_ranks.get(item_id)
        jev_rank = jev_ranks.get(item_id)
        score = 0.0
        if fused_rank is not None:
            score += 1.0 / (k + fused_rank)
        if jev_rank is not None:
            score += 1.0 / (k + jev_rank)
        scored.append((item_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def rare_exact_tokens(question: str, *, min_len: int = 5) -> List[str]:
    """Question tokens long enough to act as exact-term guards."""
    stop = {
        "about", "these", "those", "where", "which", "their", "there", "using", "based",
        "what", "when", "does", "with", "from", "that", "this", "have", "been",
    }
    tokens = tokenize(question)
    return [token for token in tokens if len(token) >= min_len and token not in stop]


def chunk_has_exact_term(text: str, terms: Sequence[str]) -> bool:
    hay = (text or "").lower()
    return any(term in hay for term in terms)


def apply_retrieval_policy(
    fused_children: List[Dict[str, Any]],
    jev_scores_by_id: Dict[str, Optional[float]],
    *,
    policy: str,
    jev_threshold: float = 0.20,
    rrf_k: int = 60,
    top_n: int = 5,
    question: str = "",
    id_key: str = "chunk_id",
) -> Dict[str, Any]:
    """Offline policies A–D over fused candidates + optional Jev scores.

    Missing Jev scores are reported; they are never fabricated.
    """
    fused = list(fused_children)
    fused_ranks = {item[id_key]: index for index, item in enumerate(fused, start=1)}
    missing = [item[id_key] for item in fused if jev_scores_by_id.get(item[id_key]) is None]
    present = {
        item_id: float(score)
        for item_id, score in jev_scores_by_id.items()
        if score is not None and item_id in fused_ranks
    }
    by_id = {item[id_key]: item for item in fused}
    terms = rare_exact_tokens(question)

    if policy in ("fused_only", "E"):
        ordered_ids = [item[id_key] for item in fused]
        kept_ids = ordered_ids[:top_n]
        return {
            "policy": policy,
            "ordered_ids": ordered_ids,
            "kept_ids": kept_ids,
            "missing_jev": missing,
            "scores": {item_id: float(by_id[item_id].get("rrf_score") or 0) for item_id in ordered_ids},
        }

    if policy == "A":
        # Hard drop below Jev threshold; order by Jev score.
        scored = sorted(present.items(), key=lambda item: item[1], reverse=True)
        kept_ids = [item_id for item_id, score in scored if score >= jev_threshold][:top_n]
        ordered_ids = [item_id for item_id, _score in scored] + [
            item_id for item_id in fused_ranks if item_id not in present
        ]
        return {
            "policy": "A",
            "ordered_ids": ordered_ids,
            "kept_ids": kept_ids,
            "missing_jev": missing,
            "scores": dict(present),
        }

    jev_ranks = {
        item_id: rank
        for rank, (item_id, _score) in enumerate(
            sorted(present.items(), key=lambda item: item[1], reverse=True),
            start=1,
        )
    }
    # Unscored items keep a weak fused-only contribution via soft RRF without a Jev term.
    blended = soft_rrf_ranks(fused_ranks, jev_ranks, k=rrf_k)
    ordered_ids = [item_id for item_id, _score in blended]

    if policy == "B":
        kept_ids = ordered_ids[:top_n]
    elif policy == "C":
        protected = [item[id_key] for item in fused[:3]]
        rest = [
            item_id
            for item_id in ordered_ids
            if item_id not in protected and present.get(item_id, 0.0) >= jev_threshold
        ]
        # Protect top-3 fused regardless of Jev; fill from thresholded rest then blend order.
        kept_ids = []
        for item_id in protected + rest + ordered_ids:
            if item_id not in kept_ids:
                kept_ids.append(item_id)
            if len(kept_ids) >= top_n:
                break
    elif policy == "D":
        # Soft RRF order, but exact-term hits are never dropped from the candidate pool
        # before taking top_n (they are boosted to stay eligible).
        guarded = {
            item_id
            for item_id in ordered_ids
            if chunk_has_exact_term(by_id[item_id].get("text") or "", terms)
        }
        # Re-order: keep soft order but ensure guarded items among top fused that match
        # are not excluded — since B already keeps without drops, D equals B plus
        # forcing guarded fused hits into the top_n window if missing.
        kept_ids = list(ordered_ids[:top_n])
        for item_id in ordered_ids:
            if item_id in guarded and item_id not in kept_ids:
                # Replace the lowest soft hit if needed.
                if len(kept_ids) >= top_n:
                    kept_ids[-1] = item_id
                else:
                    kept_ids.append(item_id)
                break
        # Stable unique
        deduped = []
        for item_id in kept_ids:
            if item_id not in deduped:
                deduped.append(item_id)
        kept_ids = deduped[:top_n]
    else:
        raise ValueError(f"Unknown retrieval policy {policy!r}")

    return {
        "policy": policy,
        "ordered_ids": ordered_ids,
        "kept_ids": kept_ids,
        "missing_jev": missing,
        "scores": {item_id: score for item_id, score in blended},
    }


_INJECTION = (
    re.compile(r"ignore (all |any |the )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (the )?(system|previous|prior)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
)


def injection_match(text: str) -> Optional[str]:
    for pattern in _INJECTION:
        found = pattern.search(text or "")
        if found:
            return found.group(0)
    return None


def passage_looks_like_instructions(text: str) -> bool:
    return injection_match(text) is not None


def split_sentences(answer: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (answer or "").strip())
    return [part.strip() for part in parts if part.strip()]


def normalize_match_text(text: str) -> str:
    """Collapse whitespace and normalize Unicode, quotes, hyphenation, and ellipses."""
    cleaned = (text or "").replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("„", '"').replace("«", '"').replace("»", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'").replace("‚", "'")
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("−", "-").replace("­", "")
    cleaned = cleaned.replace("…", "...").replace("⋯", "...")
    cleaned = cleaned.replace("％", "%").replace("﹪", "%")
    cleaned = re.sub(r"\s*\.\.\.\s*", "...", cleaned)
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_number_token(token: str) -> str:
    value = (token or "").strip().replace(",", "").replace(" ", "")
    value = value.rstrip("%").rstrip("٪")
    if value.endswith(".0") and value.replace(".", "", 1).isdigit():
        value = value[:-2]
    return value


def normalize_unsupported_indexes(raw_indexes: List[Any], sentence_count: int) -> List[int]:
    """Accept 1-based indexes; if any 0 is present, treat the list as 0-based and convert."""
    parsed: List[int] = []
    for value in raw_indexes or []:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return []
    if any(index == 0 for index in parsed):
        parsed = [index + 1 for index in parsed]
    return [index for index in parsed if 1 <= index <= max(0, sentence_count)]


def kept_sentences(answer: str, unsupported_indexes: List[int], *, minimum_chars: int) -> Dict[str, Any]:
    sentences = split_sentences(answer)
    blocked = set(unsupported_indexes)
    kept = [sentence for index, sentence in enumerate(sentences, start=1) if index not in blocked]
    text = " ".join(kept).strip()
    return {
        "text": text,
        "partially_supported": bool(sentences) and len(kept) < len(sentences) and bool(kept),
        "enough": len(text) >= minimum_chars,
        "kept_count": len(kept),
        "sentence_count": len(sentences),
    }


def corpus_contains_span(corpus: str, span: str) -> bool:
    if not span:
        return True
    if span in corpus:
        return True
    return normalize_match_text(span) in normalize_match_text(corpus)


def corpus_contains_number(corpus: str, number: str) -> bool:
    if not number:
        return True
    if number in corpus:
        return True
    plain = normalize_number_token(number)
    if plain and plain in corpus.replace(",", "").replace(" ", ""):
        return True
    normalized_corpus = normalize_match_text(corpus).replace(",", "")
    return bool(plain) and plain in normalized_corpus.replace(" ", "")


def deterministic_violation_details(
    answer: str,
    contexts: List[Dict[str, Any]],
    *,
    min_quote_chars: int,
) -> List[Dict[str, Any]]:
    """Structured fail-closed checks for citations, quotations, and numbers."""
    details: List[Dict[str, Any]] = []
    limit = len(contexts)
    # Strip citation markers before number scans so [12] is not treated as the number 12.
    answer_without_citations = re.sub(r"\[(\d+)\]", " ", answer or "")
    cited = [int(number) for number in re.findall(r"\[(\d+)\]", answer or "")]
    for number in cited:
        if number < 1 or number > limit:
            details.append(
                {
                    "kind": "citation_index",
                    "message": f"Citation [{number}] does not match a retrieved passage.",
                    "offending": f"[{number}]",
                    "passage_text": "\n".join(
                        f"[{index}] {(context.get('text') or '')[:240]}"
                        for index, context in enumerate(contexts, start=1)
                    ),
                }
            )
    corpus = "\n".join(str(context.get("text") or "") for context in contexts)
    for quote in re.findall(r"[\"“](.{8,}?)[\"”]", answer or "", flags=re.S):
        snippet = quote.strip().strip(".,;:")
        if len(snippet) >= min_quote_chars and not corpus_contains_span(corpus, snippet):
            details.append(
                {
                    "kind": "quote_span",
                    "message": "A quotation in the answer is not in the cited passages.",
                    "offending": snippet,
                    "passage_text": corpus,
                }
            )
            break
    # Also ignore bare page markers the writer often copies from the prompt header.
    answer_for_numbers = re.sub(r"\bpage\s+\d+\b", " ", answer_without_citations, flags=re.I)
    for number in re.findall(r"\d[\d,]*(?:\.\d+)?%?", answer_for_numbers):
        if re.fullmatch(r"\d%?", number):
            continue
        if not corpus_contains_number(corpus, number):
            details.append(
                {
                    "kind": "number",
                    "message": f"The number {number} is not in the cited passages.",
                    "offending": number,
                    "passage_text": corpus,
                }
            )
            break
    return details


def deterministic_violations(answer: str, contexts: List[Dict[str, Any]], *, min_quote_chars: int) -> List[str]:
    """Fail closed when citations, quotations, or numbers are not in the cited text."""
    return [item["message"] for item in deterministic_violation_details(answer, contexts, min_quote_chars=min_quote_chars)]


def assert_identical_passages(writer_passages: List[Dict[str, Any]], checker_passages: List[Dict[str, Any]]) -> None:
    writer_texts = [str(item.get("text") or "") for item in writer_passages]
    checker_texts = [str(item.get("text") or "") for item in checker_passages]
    if writer_texts != checker_texts:
        raise AssertionError("Writer and checker passages are not byte-identical for the same request.")


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    expected = [item for item in expected_ids if item]
    if not expected:
        return 1.0
    found = set(retrieved_ids[: max(0, k)])
    return len(found.intersection(expected)) / len(set(expected))


def publish_allowed(
    old_recall: float,
    new_recall: float,
    margin: float,
    *,
    golden_count: int = 0,
    min_golden: int = 30,
    override: bool = False,
) -> bool:
    """Fail closed when the golden set is too small, unless an explicit override is set."""
    if int(golden_count) < int(min_golden) and not override:
        return False
    return new_recall + 1e-9 >= old_recall - margin


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
