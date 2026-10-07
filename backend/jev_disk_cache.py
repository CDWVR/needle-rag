"""Persistent on-disk Jev score cache.

Key: (query_hash, chunk_content_hash, jev_model, prompt_version).
Never fabricates scores for missing pairs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    def get_many(
        self,
        *,
        query: str,
        chunk_texts: Sequence[str],
        jev_model: str,
        prompt_version: str = JEV_PROMPT_VERSION,
    ) -> Tuple[List[Optional[float]], List[int]]:
        scores: List[Optional[float]] = []
        missing: List[int] = []
        for index, text in enumerate(chunk_texts):
            value = self.get(query=query, chunk_text=text, jev_model=jev_model, prompt_version=prompt_version)
            scores.append(value)
            if value is None:
                missing.append(index)
        return scores, missing

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM jev_scores").fetchone()
            return int(row[0] if row else 0)


class AnswerabilityDiskCache:
    """Cache for E3 yes/no answerability judgments."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(os.path.dirname(__file__), "chroma_store", "answerability_cache.sqlite")
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS answerability (
                cache_key TEXT PRIMARY KEY,
                answerable INTEGER NOT NULL,
                raw TEXT,
                model TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def make_key(question: str, passage_texts: Sequence[str], model: str) -> str:
        blob = json.dumps(
            {"q": question, "p": list(passage_texts), "m": model},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT answerable, raw, model FROM answerability WHERE cache_key=?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return {"answerable": bool(row[0]), "raw": row[1], "model": row[2]}

    def put(self, key: str, *, answerable: bool, raw: str, model: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO answerability (cache_key, answerable, raw, model)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    answerable=excluded.answerable,
                    raw=excluded.raw,
                    model=excluded.model
                """,
                (key, 1 if answerable else 0, raw, model),
            )
            self._conn.commit()
