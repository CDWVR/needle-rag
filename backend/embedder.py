"""Embedding backends. Local MiniLM by default; BGE-M3 via OpenRouter or TEI when EMBED_BACKEND says so.

Every index version records the model that built it. Queries against a version are embedded
with that same model, so comparing an old and a new index (the refresh gate) is meaningful
even when the model changed between them.
"""

import os
import threading
from typing import Callable, Dict, List, Optional

from config import EMBEDDING_MODEL_ID
from pipeline_logic import IncompatibleIndex

_clients: Dict[str, object] = {}
_lock = threading.Lock()


def _backend() -> str:
    return (os.getenv("EMBED_BACKEND", "minilm") or "minilm").strip().lower()


def _openrouter_model() -> str:
    from embeddings.bge_m3_openrouter import BGE_M3_MODEL

    return BGE_M3_MODEL


def _tei_model() -> str:
    from embeddings.tei_client import BGE_M3_MODEL

    return BGE_M3_MODEL


def configured_embedding_model() -> str:
    """Model id the configured backend produces; new documents are embedded with it."""
    backend = _backend()
    if backend == "openrouter":
        return _openrouter_model()
    if backend == "tei":
        return _tei_model()
    return EMBEDDING_MODEL_ID


def embedding_dimensions() -> int:
    return 1024 if _backend() in {"openrouter", "tei"} else 384


def _client(name: str, factory: Callable[[], object]):
    with _lock:
        if name not in _clients:
            _clients[name] = factory()
        return _clients[name]


def _minilm(texts: List[str]) -> List[List[float]]:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    function = _client("minilm", DefaultEmbeddingFunction)
    return [[float(value) for value in vector] for vector in function(texts)]


def _openrouter(texts: List[str]) -> List[List[float]]:
    from embeddings import BgeM3EmbeddingClient

    return _client("openrouter", BgeM3EmbeddingClient).embed(texts)


def _tei(texts: List[str]) -> List[List[float]]:
    from embeddings import TeiEmbeddingClient

    return _client("tei", TeiEmbeddingClient).embed(texts)


def _backend_for(model_id: str) -> Callable[[List[str]], List[List[float]]]:
    if model_id == EMBEDDING_MODEL_ID:
        return _minilm
    if model_id == _openrouter_model():
        return _openrouter
    if model_id == _tei_model():
        return _tei
    raise IncompatibleIndex(f"No embedding backend is configured for {model_id}. Refresh the index.")


def embed(texts: List[str], model_id: Optional[str] = None) -> List[List[float]]:
    """Embed with `model_id` (defaults to the configured model)."""
    if not texts:
        return []
    return _backend_for(model_id or configured_embedding_model())(texts)
