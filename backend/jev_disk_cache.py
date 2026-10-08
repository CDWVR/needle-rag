"""Persistent on-disk Jev score cache.

Key: (query_hash, chunk_content_hash, jev_model, prompt_version).
Never fabricates scores for missing pairs.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from typing import Optional

from paths import data_path

JEV_PROMPT_VERSION = os.getenv("JEV_PROMPT_VERSION", "relevance_rerank_v1").strip() or "relevance_rerank_v1"

DEFAULT_PATH = data_path("chroma_store", "jev_score_cache.sqlite")


def query_hash(query: str) -> str:
    normalized = " ".join((query or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def chunk_content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class JevDiskCache:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jev_scores (
                query_hash TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                jev_model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (query_hash, chunk_hash, jev_model, prompt_version)
            )
            """
        )
        self._conn.commit()
        self.stats = {"hits": 0, "misses": 0, "writes": 0}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get(
        self,
        *,
        query: str,
        chunk_text: str,
        jev_model: str,
        prompt_version: str = JEV_PROMPT_VERSION,
    ) -> Optional[float]:
        key = (query_hash(query), chunk_content_hash(chunk_text), jev_model, prompt_version)
        with self._lock:
            row = self._conn.execute(
                "SELECT score FROM jev_scores WHERE query_hash=? AND chunk_hash=? AND jev_model=? AND prompt_version=?",
                key,
            ).fetchone()
            if row is None:
                self.stats["misses"] += 1
                return None
            self.stats["hits"] += 1
            return float(row[0])

    def put(
        self,
        *,
        query: str,
        chunk_text: str,
        jev_model: str,
        score: float,
        prompt_version: str = JEV_PROMPT_VERSION,
    ) -> None:
        key = (query_hash(query), chunk_content_hash(chunk_text), jev_model, prompt_version, float(score))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jev_scores (query_hash, chunk_hash, jev_model, prompt_version, score)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(query_hash, chunk_hash, jev_model, prompt_version)
                DO UPDATE SET score=excluded.score
                """,
                key,
            )
            self._conn.commit()
            self.stats["writes"] += 1

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM jev_scores").fetchone()
            return int(row[0] if row else 0)

