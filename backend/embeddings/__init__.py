"""Pluggable embedding backends. Production default remains local MiniLM."""

from embeddings.base import EmbeddingClient, L2_normalize
from embeddings.minilm_local import MiniLMEmbeddingClient
from embeddings.bge_m3_openrouter import BgeM3EmbeddingClient, BGE_M3_MODEL, BGE_M3_PRICE_PER_M
from embeddings.tei_client import TeiEmbeddingClient
from embeddings.routing import RoutingBgeM3Client, assert_cross_backend_cosine, cosine

__all__ = [
    "EmbeddingClient",
    "L2_normalize",
    "MiniLMEmbeddingClient",
    "BgeM3EmbeddingClient",
    "TeiEmbeddingClient",
    "RoutingBgeM3Client",
    "assert_cross_backend_cosine",
    "cosine",
    "BGE_M3_MODEL",
    "BGE_M3_PRICE_PER_M",
]
