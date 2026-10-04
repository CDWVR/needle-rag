"""Local MiniLM embeddings via Chroma's DefaultEmbeddingFunction (already installed)."""

from __future__ import annotations

from typing import List, Sequence

from embeddings.base import L2_normalize


class MiniLMEmbeddingClient:
    model_id = "all-MiniLM-L6-v2"
    dimensions = 384

    def __init__(self):
        self._fn = None

    def _function(self):
        if self._fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._fn = DefaultEmbeddingFunction()
        return self._fn

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._function()(list(texts))
        return [L2_normalize(vector) for vector in vectors]
