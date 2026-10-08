"""Remote BGE-M3 embedding clients (OpenRouter or a TEI server). Local MiniLM lives in embedder.py."""

from embeddings.base import EmbeddingClient, L2_normalize
from embeddings.bge_m3_openrouter import BGE_M3_MODEL, BgeM3EmbeddingClient
from embeddings.tei_client import TeiEmbeddingClient

__all__ = ["BGE_M3_MODEL", "BgeM3EmbeddingClient", "EmbeddingClient", "L2_normalize", "TeiEmbeddingClient"]
