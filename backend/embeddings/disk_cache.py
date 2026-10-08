"""Persistent embedding cache keyed by (model, text hash)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from typing import List, Optional, Sequence

from paths import data_path

DEFAULT_PATH = data_path("chroma_store", "embedding_cache.sqlite")


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class EmbeddingDiskCache:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                dims INTEGER NOT NULL,
                vector TEXT NOT NULL,
                PRIMARY KEY (model, text_hash)
            )
            """
        )
        self._conn.commit()
        self.stats = {"hits": 0, "misses": 0, "writes": 0}

    def get(self, model: str, text: str) -> Optional[List[float]]:
        key = (model, text_hash(text))
        with self._lock:
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE model=? AND text_hash=?",
                key,
            ).fetchone()
            if not row:
                self.stats["misses"] += 1
                return None
            self.stats["hits"] += 1
            return list(json.loads(row[0]))

    def put(self, model: str, text: str, vector: Sequence[float]) -> None:
        payload = json.dumps([float(value) for value in vector])
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO embeddings (model, text_hash, dims, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model, text_hash) DO UPDATE SET
                    dims=excluded.dims,
                    vector=excluded.vector
                """,
                (model, text_hash(text), len(vector), payload),
            )
            self._conn.commit()
            self.stats["writes"] += 1

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return int(row[0] if row else 0)
