"""BGE-M3 dense embeddings via OpenRouter's OpenAI-compatible API.

No local weights. Production defaults are unchanged; select with EMBED_BACKEND=openrouter.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from embeddings.base import L2_normalize
from embeddings.disk_cache import EmbeddingDiskCache

log = logging.getLogger("needle.embeddings.bge_m3")

BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "baai/bge-m3").strip() or "baai/bge-m3"
BGE_M3_DIMS = 1024
# OpenRouter listed price: $0.01 / M input tokens (prompt=1e-8 USD/token).
BGE_M3_PRICE_PER_M = float(os.getenv("BGE_M3_PRICE_PER_M", "0.01"))
OPENROUTER_EMBED_URL = os.getenv(
    "OPENROUTER_EMBEDDINGS_URL",
    "https://openrouter.ai/api/v1/embeddings",
).strip()
# Prefer zero-data-retention routing when providers advertise it.
ZDR_PROVIDER_ORDER = [
    part.strip()
    for part in os.getenv("BGE_M3_ZDR_PROVIDERS", "Parasail,DeepInfra").split(",")
    if part.strip()
]


class BgeM3EmbeddingClient:
    """Dense-only BGE-M3 client (1024-d, L2-normalized, no query instruction)."""

    model_id = BGE_M3_MODEL
    dimensions = BGE_M3_DIMS

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        max_input_tokens: int = 512,
        batch_size: int = 32,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        cache: Optional[EmbeddingDiskCache] = None,
        require_zdr: bool = True,
    ):
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for BgeM3EmbeddingClient.")
        self.max_input_tokens = max(32, int(os.getenv("BGE_M3_MAX_INPUT_TOKENS", str(max_input_tokens))))
        self.batch_size = max(1, int(os.getenv("BGE_M3_BATCH_SIZE", str(batch_size))))
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.cache = cache if cache is not None else EmbeddingDiskCache()
        self.require_zdr = require_zdr
        self.last_providers: List[str] = []
        self.stats = {
            "requests": 0,
            "tokens": 0,
            "cost_usd": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @staticmethod
    def verify_model_listing() -> Dict[str, Any]:
        """Fetch OpenRouter endpoints metadata for the configured slug."""
        url = f"https://openrouter.ai/api/v1/models/{BGE_M3_MODEL}/endpoints"
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
        response.raise_for_status()
        payload = response.json().get("data") or {}
        endpoints = payload.get("endpoints") or []
        prices = []
        providers = []
        for endpoint in endpoints:
            pricing = endpoint.get("pricing") or {}
            prompt = float(pricing.get("prompt") or 0)
            prices.append(prompt * 1_000_000)
            providers.append(
                {
                    "provider_name": endpoint.get("provider_name"),
                    "tag": endpoint.get("tag"),
                    "price_per_m_tokens": prompt * 1_000_000,
                    "context_length": endpoint.get("context_length"),
                }
            )
        return {
            "slug": payload.get("id") or BGE_M3_MODEL,
            "name": payload.get("name"),
            "modality": (payload.get("architecture") or {}).get("modality"),
            "price_per_m_tokens_usd": min(prices) if prices else BGE_M3_PRICE_PER_M,
            "providers": providers,
        }

    def estimate_cost_usd(self, texts: Sequence[str]) -> float:
        # Conservative: 1 token ≈ 4 chars, clamp each input to max_input_tokens.
        tokens = 0
        for text in texts:
            approx = max(1, (len(text or "") + 3) // 4)
            tokens += min(approx, self.max_input_tokens)
        return (tokens / 1_000_000.0) * BGE_M3_PRICE_PER_M

    def _truncate(self, text: str) -> str:
        # Approximate token clamp without a local tokenizer download.
        max_chars = self.max_input_tokens * 4
        value = text or ""
        if len(value) <= max_chars:
            return value
        return value[:max_chars]

    def _provider_payload(self) -> Dict[str, Any]:
        provider: Dict[str, Any] = {
            "allow_fallbacks": True,
            "order": list(ZDR_PROVIDER_ORDER),
        }
        if self.require_zdr:
            # OpenRouter: deny providers that retain prompts for training.
            provider["data_collection"] = "deny"
        return provider

    def _request_batch(self, texts: Sequence[str]) -> tuple[List[List[float]], str, int]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost/needle-rag"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "needle-rag-eval"),
        }
        body = {
            "model": self.model_id,
            "input": list(texts),
            "encoding_format": "float",
            "provider": self._provider_payload(),
        }
        delay = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(OPENROUTER_EMBED_URL, headers=headers, json=body)
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After") or delay)
                    log.warning("BGE-M3 rate limited (429); sleeping %.1fs", retry_after)
                    time.sleep(retry_after)
                    delay = min(delay * 2, 30)
                    continue
                if response.status_code >= 500:
                    log.warning("BGE-M3 upstream %s; retrying", response.status_code)
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if response.status_code != 200:
                    raise RuntimeError(
                        f"OpenRouter embeddings HTTP {response.status_code}: {response.text[:300]}"
                    )
                payload = response.json()
                provider = (
                    response.headers.get("x-openrouter-provider")
                    or response.headers.get("X-OpenRouter-Provider")
                    or (payload.get("provider") if isinstance(payload.get("provider"), str) else None)
                    or "unknown"
                )
                rows = sorted(payload.get("data") or [], key=lambda item: int(item.get("index") or 0))
                vectors = [list(item.get("embedding") or []) for item in rows]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Embedding count mismatch: got {len(vectors)} for {len(texts)} inputs"
                    )
                for vector in vectors:
                    if len(vector) != self.dimensions:
                        raise RuntimeError(
                            f"Expected {self.dimensions}-d dense vector, got {len(vector)}"
                        )
                usage = payload.get("usage") or {}
                tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
                return vectors, str(provider), tokens
            except (httpx.TimeoutException, httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                log.warning("BGE-M3 embed attempt %s failed: %s", attempt + 1, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError(f"BGE-M3 embedding failed after retries: {last_error}")

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        outputs: List[Optional[List[float]]] = [None] * len(texts)
        pending_indexes: List[int] = []
        pending_texts: List[str] = []
        for index, text in enumerate(texts):
            clipped = self._truncate(text)
            cached = self.cache.get(self.model_id, clipped)
            if cached is not None:
                outputs[index] = L2_normalize(cached)
                self.stats["cache_hits"] += 1
            else:
                pending_indexes.append(index)
                pending_texts.append(clipped)
                self.stats["cache_misses"] += 1

        for start in range(0, len(pending_texts), self.batch_size):
            batch_texts = pending_texts[start : start + self.batch_size]
            batch_indexes = pending_indexes[start : start + self.batch_size]
            vectors, provider, tokens = self._request_batch(batch_texts)
            self.last_providers.append(provider)
            log.info("BGE-M3 batch served by provider=%s n=%s tokens=%s", provider, len(batch_texts), tokens)
            self.stats["requests"] += 1
            self.stats["tokens"] += tokens
            cost = (tokens / 1_000_000.0) * BGE_M3_PRICE_PER_M
            self.stats["cost_usd"] = round(self.stats["cost_usd"] + cost, 6)
            for offset, vector in enumerate(vectors):
                normalized = L2_normalize(vector)
                original_index = batch_indexes[offset]
                outputs[original_index] = normalized
                self.cache.put(self.model_id, batch_texts[offset], normalized)

        return [list(vector or []) for vector in outputs]
