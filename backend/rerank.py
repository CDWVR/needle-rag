"""Jev reranking through OpenRouter, its score caches, and the local fallback rankers."""

import logging
import os
import threading
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from config import (
    CIRCUIT_FAILURES,
    CIRCUIT_RESET_SECONDS,
    JEV_CACHE_TTL_SECONDS,
    JEV_COST_PER_CANDIDATE,
    JEV_ENDPOINT,
    JEV_MAX_CONCURRENCY,
    JEV_MAX_RETRIES,
    JEV_MODEL,
    JEV_TIMEOUT_SECONDS,
)
from llm import record_cost
from pipeline_logic import (
    CircuitBreaker,
    JevNotConfigured,
    JevUnavailable,
    NeedleError,
    ScoreCache,
    fused_fallback_scores,
    normalize_query,
)

log = logging.getLogger("needle")
_jev_score_cache = ScoreCache(JEV_CACHE_TTL_SECONDS)
jev_breaker = CircuitBreaker(CIRCUIT_FAILURES, CIRCUIT_RESET_SECONDS)
_last_rerank_mode = "jev"


def last_rerank_mode() -> str:
    """The ranker the most recent question used (shown on the Pipeline page)."""
    return _last_rerank_mode


def note_rerank_mode(mode: str) -> None:
    global _last_rerank_mode
    _last_rerank_mode = mode


_jev_cache_enabled = True
_jev_cache_stats = {"hits": 0, "misses": 0}
_reuse_jev_disk_cache = False
_jev_disk_cache = None


def set_reuse_jev_cache(enabled: bool) -> None:
    """When True, only disk-cached Jev scores are used; missing pairs are not fetched."""
    global _reuse_jev_disk_cache, _jev_disk_cache
    _reuse_jev_disk_cache = bool(enabled)
    if _reuse_jev_disk_cache and _jev_disk_cache is None:
        from jev_disk_cache import JevDiskCache

        _jev_disk_cache = JevDiskCache()


def enable_jev_disk_cache(enabled: bool = True) -> None:
    global _jev_disk_cache
    if enabled:
        from jev_disk_cache import JevDiskCache

        if _jev_disk_cache is None:
            _jev_disk_cache = JevDiskCache()
    else:
        _jev_disk_cache = None


def jev_cache_stats() -> Dict[str, Any]:
    hits = int(_jev_cache_stats["hits"])
    misses = int(_jev_cache_stats["misses"])
    total = hits + misses
    disk = None
    if _jev_disk_cache is not None:
        disk = {"path": _jev_disk_cache.path, "rows": _jev_disk_cache.count(), **_jev_disk_cache.stats}
    return {
        "enabled": _jev_cache_enabled,
        "reuse_disk": _reuse_jev_disk_cache,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "disk": disk,
    }


def jev_configured() -> bool:
    try:
        jev_api_key()
    except JevNotConfigured:
        return False
    return True


def jev_api_key() -> str:
    """Jev is reached through OpenRouter, so it uses the same key as the answer models."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise JevNotConfigured("Set OPENROUTER_API_KEY in .env to enable Jev reranking.")
    return api_key


_jev_reranker = None
_jev_reranker_signature = None
_jev_reranker_lock = threading.Lock()


def _jev_client():
    """One reranker per configuration; building it loads a tokenizer, so do it once."""
    global _jev_reranker, _jev_reranker_signature
    try:
        from jev_reranker import JevReranker
    except ImportError as exc:
        raise JevUnavailable("The jev-reranker package is not installed.") from exc
    api_key = jev_api_key()
    signature = (api_key, JEV_MODEL, JEV_ENDPOINT, JEV_MAX_CONCURRENCY, JEV_TIMEOUT_SECONDS, JEV_MAX_RETRIES)
    with _jev_reranker_lock:
        if _jev_reranker is None or _jev_reranker_signature != signature:
            _jev_reranker = JevReranker(
                api_key=api_key,
                model=JEV_MODEL,
                endpoint=JEV_ENDPOINT,
                mode="pointwise",
                max_concurrency=max(1, JEV_MAX_CONCURRENCY),
                timeout=max(1.0, JEV_TIMEOUT_SECONDS),
                max_retries=max(0, JEV_MAX_RETRIES),
                dotenv_path=None,
            )
            _jev_reranker_signature = signature
        return _jev_reranker


def _jev_scores(query: str, passages: List[str]) -> List[float]:
    if not passages:
        return []
    reranker = _jev_client()
    try:
        response = reranker.relevance_rerank(query, passages, threshold=0.0)
    except NeedleError:
        raise
    except Exception as exc:
        log.exception("Jev rerank failed")
        raise JevUnavailable("Jev could not rerank the retrieved passages.") from exc
    scores = [0.0] * len(passages)
    for item in response.get("results") or []:
        index = int(item.get("document_index", -1))
        if 0 <= index < len(scores):
            scores[index] = float(item.get("score") or 0)
    return scores


def _rerank_with_jev(query: str, candidates: List[Dict[str, Any]], version_id: str) -> tuple:
    """Return (ranked, number of scores fetched from Jev). Cached scores are free."""
    normalized = normalize_query(query)
    scores: Dict[int, float] = {}
    misses: List[int] = []
    missing_disk: List[int] = []
    for index, candidate in enumerate(candidates):
        cached = None
        if _jev_disk_cache is not None:
            cached = _jev_disk_cache.get(query=query, chunk_text=candidate.get("text") or "", jev_model=JEV_MODEL)
        if cached is None and _jev_cache_enabled:
            cached = _jev_score_cache.get((normalized, candidate["chunk_id"], version_id))
        if cached is None:
            misses.append(index)
            _jev_cache_stats["misses"] += 1
            if _reuse_jev_disk_cache:
                missing_disk.append(index)
        else:
            scores[index] = cached
            _jev_cache_stats["hits"] += 1
    if missing_disk and _reuse_jev_disk_cache:
        # Offline reuse: never fabricate; return only cached rows and mark the rest missing.
        ranked = [{"document_index": index, "score": score} for index, score in scores.items()]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked, 0
    fetched = 0
    if misses and not _reuse_jev_disk_cache:
        fresh = _jev_scores(query, [candidates[index]["text"] for index in misses])
        fetched = len(misses)
        record_cost("jev", fetched * JEV_COST_PER_CANDIDATE)
        for offset, score in enumerate(fresh):
            original = misses[offset]
            if _jev_cache_enabled:
                _jev_score_cache.put((normalized, candidates[original]["chunk_id"], version_id), score)
            if _jev_disk_cache is not None:
                _jev_disk_cache.put(
                    query=query,
                    chunk_text=candidates[original].get("text") or "",
                    jev_model=JEV_MODEL,
                    score=score,
                )
            scores[original] = score
    ranked = [{"document_index": index, "score": score} for index, score in scores.items()]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked, fetched


_cross_encoder = None


def _cross_encoder_scores(query: str, passages: List[str]) -> Optional[List[float]]:
    global _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    try:
        if _cross_encoder is None:
            # Loading the weights takes seconds; keep one instance for the process.
            model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            _cross_encoder = CrossEncoder(model_name)
        raw_scores = _cross_encoder.predict([(query, passage) for passage in passages])
    except Exception:
        log.exception("Local cross-encoder failed")
        return None
    scores = []
    for value in raw_scores:
        number = float(value)
        scores.append(1.0 / (1.0 + pow(2.718281828, -number)))
    return scores


def _fallback_scores(query: str, candidates: List[Dict[str, Any]]) -> tuple:
    local = _cross_encoder_scores(query, [candidate["text"] for candidate in candidates])
    if local is not None:
        ranked = [{"document_index": index, "score": score} for index, score in enumerate(local)]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked, "cross-encoder"
    ranked = [
        {"document_index": index, "score": score}
        for index, score in enumerate(fused_fallback_scores(len(candidates)))
    ]
    return ranked, "fused"


def rank_candidates(query: str, candidates: List[Dict[str, Any]], version_id: str) -> tuple:
    """Return (ranked, ranker name, Jev scores fetched). Falls back when Jev is down or not configured."""
    if jev_breaker.closed():
        try:
            ranked, fetched = _rerank_with_jev(query, candidates, version_id)
            jev_breaker.success()
            return ranked, "jev", fetched
        except JevNotConfigured as exc:
            # A missing key is configuration, not an outage: do not trip the breaker.
            log.warning("%s Using the fallback ranker.", exc)
        except Exception:
            log.exception("Jev failed; using the fallback ranker")
            jev_breaker.failure()
    ranked, mode = _fallback_scores(query, candidates)
    return ranked, mode, 0
