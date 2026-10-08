"""Shared embedding client interface."""

from __future__ import annotations

import math
from typing import List, Protocol, Sequence


def L2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


class EmbeddingClient(Protocol):
    model_id: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return one dense vector per input text."""
        ...
