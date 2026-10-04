"""Minimal TEI-compatible HTTP server for BGE-M3 when Docker is unavailable.

Speaks /health and /embed like text-embeddings-inference.
Weights are cached under TEI_MODEL_CACHE (default: ./tei_model_cache).

Usage:
  pip install sentence-transformers
  python -m embeddings.tei_local_server
"""

from __future__ import annotations

import os
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

MODEL_ID = os.getenv("TEI_MODEL_ID", "BAAI/bge-m3")
CACHE_DIR = os.getenv(
    "TEI_MODEL_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "tei_model_cache"),
)
HOST = os.getenv("TEI_HOST", "127.0.0.1")
PORT = int(os.getenv("TEI_PORT", "18080"))

app = FastAPI(title="needle-tei-local", version="0.1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_ID, cache_folder=CACHE_DIR)
    return _model


class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]
    normalize: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/embed")
def embed(body: EmbedRequest):
    texts = [body.inputs] if isinstance(body.inputs, str) else list(body.inputs)
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=bool(body.normalize),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [row.tolist() for row in vectors]


def main() -> None:
    # Warm on startup so first /embed is not dominated by model load.
    get_model()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
