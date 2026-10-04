"""BGE-M3 embeddings via a TEI (text-embeddings-inference) HTTP endpoint.

Selected with EMBED_BACKEND=tei. Does not change the MiniLM production default.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Sequence

import httpx

from embeddings.base import L2_normalize
from embeddings.disk_cache import EmbeddingDiskCache

log = logging.getLogger("needle.embeddings.tei")

BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3").strip() or "BAAI/bge-m3"
BGE_M3_DIMS = 1024
DEFAULT_TEI_URL = os.getenv("TEI_URL", "http://127.0.0.1:18080").rstrip("/")


class TeiEmbeddingClient:
    """Dense-only BGE-M3 client against TEI /embed (or OpenAI-compatible /embeddings)."""

    model_id = BGE_M3_MODEL
    dimensions = BGE_M3_DIMS

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        max_input_tokens: int = 512,
        batch_size: int = 32,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        cache: Optional[EmbeddingDiskCache] = None,
    ):
        self.base_url = (base_url or DEFAULT_TEI_URL).rstrip("/")
        self.max_input_tokens = max(32, int(os.getenv("BGE_M3_MAX_INPUT_TOKENS", str(max_input_tokens))))
        self.batch_size = max(1, int(os.getenv("TEI_BATCH_SIZE", str(batch_size))))
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.cache = cache if cache is not None else EmbeddingDiskCache()
        self.stats = {
            "requests": 0,
            "tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "embed_ms": 0.0,
            "chunks": 0,
        }

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def _truncate(self, text: str) -> str:
        max_chars = self.max_input_tokens * 4
        value = text or ""
        return value if len(value) <= max_chars else value[:max_chars]

    def _request_batch(self, texts: Sequence[str]) -> List[List[float]]:
        delay = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    # Prefer native TEI /embed; fall back to OpenAI-compatible /v1/embeddings.
                    response = client.post(
                        f"{self.base_url}/embed",
                        json={"inputs": list(texts), "normalize": True},
                    )
                    if response.status_code == 404:
                        response = client.post(
                            f"{self.base_url}/embeddings",
                            json={"model": self.model_id, "input": list(texts)},
                        )
                if response.status_code == 429:
                    time.sleep(float(response.headers.get("Retry-After") or delay))
                    delay = min(delay * 2, 30)
                    continue
                if response.status_code >= 500:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if response.status_code != 200:
                    raise RuntimeError(f"TEI HTTP {response.status_code}: {response.text[:300]}")
                payload = response.json()
                if isinstance(payload, list):
                    vectors = payload
                else:
                    rows = sorted(payload.get("data") or [], key=lambda item: int(item.get("index") or 0))
                    vectors = [list(item.get("embedding") or []) for item in rows]
                if len(vectors) != len(texts):
                    raise RuntimeError(f"TEI count mismatch: {len(vectors)} vs {len(texts)}")
                for vector in vectors:
                    if len(vector) != self.dimensions:
                        raise RuntimeError(f"Expected {self.dimensions}-d, got {len(vector)}")
                return [[float(value) for value in vector] for vector in vectors]
            except (httpx.TimeoutException, httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                log.warning("TEI embed attempt %s failed: %s", attempt + 1, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError(f"TEI embedding failed after retries: {last_error}")

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        outputs: List[Optional[List[float]]] = [None] * len(texts)
        pending_indexes: List[int] = []
        pending_texts: List[str] = []
        for index, text in enumerate(texts):
            clipped = self._truncate(text)
            cached = self.cache.get(f"tei:{self.model_id}", clipped)
            if cached is not None:
                outputs[index] = L2_normalize(cached)
                self.stats["cache_hits"] += 1
            else:
                pending_indexes.append(index)
                pending_texts.append(clipped)
                self.stats["cache_misses"] += 1

        started = time.perf_counter()
        for start in range(0, len(pending_texts), self.batch_size):
            batch_texts = pending_texts[start : start + self.batch_size]
            batch_indexes = pending_indexes[start : start + self.batch_size]
            vectors = self._request_batch(batch_texts)
            self.stats["requests"] += 1
            self.stats["chunks"] += len(batch_texts)
            for offset, vector in enumerate(vectors):
                normalized = L2_normalize(vector)
                original_index = batch_indexes[offset]
                outputs[original_index] = normalized
                self.cache.put(f"tei:{self.model_id}", batch_texts[offset], normalized)
        self.stats["embed_ms"] += (time.perf_counter() - started) * 1000
        return [list(vector or []) for vector in outputs]
