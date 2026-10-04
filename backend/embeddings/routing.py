"""Route chunk texts to OpenRouter (safe) or TEI (sensitive) for BGE-M3 evals."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from embeddings.base import L2_normalize
from embeddings.bge_m3_openrouter import BgeM3EmbeddingClient
from embeddings.tei_client import TeiEmbeddingClient


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


class RoutingBgeM3Client:
    """Embed with TEI by default; OpenRouter only for explicitly safe texts.

    Callers pass parallel `backends` markers via embed_routed(), or set
    default_backend ('tei'|'openrouter').
    """

    model_id = "baai/bge-m3"
    dimensions = 1024

    def __init__(
        self,
        *,
        tei: Optional[TeiEmbeddingClient] = None,
        openrouter: Optional[BgeM3EmbeddingClient] = None,
        default_backend: str = "tei",
    ):
        self.tei = tei or TeiEmbeddingClient()
        self.openrouter = openrouter
        self.default_backend = default_backend
        self.last_routing: Dict[str, int] = {"tei": 0, "openrouter": 0}

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        backends = [self.default_backend] * len(texts)
        return self.embed_routed(texts, backends)

    def embed_routed(self, texts: Sequence[str], backends: Sequence[str]) -> List[List[float]]:
        if len(texts) != len(backends):
            raise ValueError("texts/backends length mismatch")
        outputs: List[Optional[List[float]]] = [None] * len(texts)
        buckets: Dict[str, List[Tuple[int, str]]] = {"tei": [], "openrouter": []}
        for index, (text, backend) in enumerate(zip(texts, backends)):
            key = "openrouter" if backend == "openrouter" else "tei"
            buckets[key].append((index, text))

        if buckets["tei"]:
            vectors = self.tei.embed([text for _i, text in buckets["tei"]])
            for (index, _text), vector in zip(buckets["tei"], vectors):
                outputs[index] = vector
            self.last_routing["tei"] += len(buckets["tei"])

        if buckets["openrouter"]:
            if self.openrouter is None:
                self.openrouter = BgeM3EmbeddingClient()
            vectors = self.openrouter.embed([text for _i, text in buckets["openrouter"]])
            for (index, _text), vector in zip(buckets["openrouter"], vectors):
                outputs[index] = vector
            self.last_routing["openrouter"] += len(buckets["openrouter"])

        return [list(vector or []) for vector in outputs]


def assert_cross_backend_cosine(
    tei: TeiEmbeddingClient,
    openrouter: BgeM3EmbeddingClient,
    texts: Sequence[str],
    *,
    min_cosine: float = 0.99,
) -> Dict[str, float]:
    """Assert mean pairwise cosine(TEI, OpenRouter) > min_cosine on the sample."""
    sample = list(texts)
    if not sample:
        return {"n": 0, "mean_cosine": 1.0, "min_cosine": 1.0, "passed": True}
    left = tei.embed(sample)
    right = openrouter.embed(sample)
    scores = [cosine(L2_normalize(a), L2_normalize(b)) for a, b in zip(left, right)]
    mean = sum(scores) / len(scores)
    lowest = min(scores)
    passed = mean > min_cosine
    if not passed:
        raise RuntimeError(
            f"Cross-backend cosine check failed: mean={mean:.4f} min={lowest:.4f} "
            f"(need mean > {min_cosine}) on n={len(scores)}"
        )
    return {
        "n": len(scores),
        "mean_cosine": round(mean, 6),
        "min_cosine": round(lowest, 6),
        "passed": True,
    }
