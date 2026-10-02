"""Versioned index registry, parent store, and deletion records.

Searches only read the active index. A refresh builds a new collection and
publishes it after the copy is verified. Deletes propagate to every version
still recorded in the registry.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


LEGACY_COLLECTION = "documind_sota_local"
LEGACY_VERSION_ID = "legacy-1"


class IndexStore:
    def __init__(self, sqlite_path: str):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                document_id UNINDEXED,
                chunk_id UNINDEXED,
                document_name UNINDEXED,
                page_number UNINDEXED,
                header_context UNINDEXED,
                parent_text,
                tokenize="porter"
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_registry (
                version_id TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                activated_at TEXT,
                embed_style TEXT NOT NULL DEFAULT 'raw',
                chunking TEXT NOT NULL DEFAULT 'Parent-child'
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parents (
                parent_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER,
                header_context TEXT,
                parent_text TEXT NOT NULL,
                summary TEXT,
                version_id TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog (
                document_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                num_pages INTEGER,
                num_chunks INTEGER,
                file_type TEXT,
                uploaded_at TEXT,
                version_id TEXT,
                status TEXT NOT NULL DEFAULT 'active'
            )
            """
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(index_registry)")}
        if "embed_style" not in columns:
            self.conn.execute("ALTER TABLE index_registry ADD COLUMN embed_style TEXT NOT NULL DEFAULT 'raw'")
        if "chunking" not in columns:
            self.conn.execute("ALTER TABLE index_registry ADD COLUMN chunking TEXT NOT NULL DEFAULT 'Parent-child'")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def ensure_active_version(self, embedding_model: str) -> Dict[str, Any]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM index_registry WHERE status = 'active' ORDER BY activated_at DESC LIMIT 1"
            ).fetchone()
            if row:
                return dict(row)
            now = _now()
            self.conn.execute(
                """
                INSERT INTO index_registry
                    (version_id, embedding_model, collection_name, status, created_at, activated_at)
                VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (LEGACY_VERSION_ID, embedding_model, LEGACY_COLLECTION, now, now),
            )
            self.conn.commit()
            return {
                "version_id": LEGACY_VERSION_ID,
                "embedding_model": embedding_model,
                "collection_name": LEGACY_COLLECTION,
                "status": "active",
                "created_at": now,
                "activated_at": now,
                "embed_style": "raw",
                "chunking": "Parent-child",
            }

    def list_versions(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM index_registry ORDER BY created_at ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def begin_version(
        self,
        version_id: str,
        embedding_model: str,
        collection_name: str,
        embed_style: str = "raw",
        chunking: str = "Parent-child",
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO index_registry
                    (version_id, embedding_model, collection_name, status, created_at, activated_at, embed_style, chunking)
                VALUES (?, ?, ?, 'building', ?, NULL, ?, ?)
                """,
                (version_id, embedding_model, collection_name, _now(), embed_style, chunking),
            )
            self.conn.commit()

    def rollback_active(self) -> Optional[Dict[str, Any]]:
        """Point search back at the newest retired index without deleting data."""
        with self._lock:
            active = self.conn.execute(
                "SELECT version_id FROM index_registry WHERE status = 'active' ORDER BY activated_at DESC LIMIT 1"
            ).fetchone()
            retired = self.conn.execute(
                "SELECT version_id FROM index_registry WHERE status = 'retired' ORDER BY activated_at DESC LIMIT 1"
            ).fetchone()
            if not active or not retired:
                return None
            now = _now()
            self.conn.execute(
                "UPDATE index_registry SET status = 'retired' WHERE version_id = ?",
                (active["version_id"],),
            )
            self.conn.execute(
                "UPDATE index_registry SET status = 'active', activated_at = ? WHERE version_id = ?",
                (now, retired["version_id"]),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT * FROM index_registry WHERE version_id = ?",
                (retired["version_id"],),
            ).fetchone()
        return dict(row) if row else None

    def publish_version(self, version_id: str) -> None:
        """Atomically retire the current active index and publish the new one."""
        with self._lock:
            now = _now()
            self.conn.execute(
                "UPDATE index_registry SET status = 'retired' WHERE status = 'active'"
            )
            updated = self.conn.execute(
                """
                UPDATE index_registry
                SET status = 'active', activated_at = ?
                WHERE version_id = ? AND status = 'building'
                """,
                (now, version_id),
            )
            if updated.rowcount != 1:
                self.conn.rollback()
                raise RuntimeError("Versioned index handoff failed before publish.")
            self.conn.execute(
                "UPDATE parents SET version_id = ? WHERE version_id IS NULL OR version_id != ?",
                (version_id, version_id),
            )
            self.conn.execute(
                "UPDATE catalog SET version_id = ? WHERE status = 'active'",
                (version_id,),
            )
            self.conn.commit()

    def fail_version(self, version_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE index_registry SET status = 'failed' WHERE version_id = ? AND status = 'building'",
                (version_id,),
            )
            self.conn.commit()

    def save_document(
        self,
        *,
        document_id: str,
        name: str,
        num_pages: int,
        num_chunks: int,
        file_type: str,
        uploaded_at: str,
        version_id: str,
        parents: List[Dict[str, Any]],
        fts_rows: List[tuple],
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO catalog
                    (document_id, name, num_pages, num_chunks, file_type, uploaded_at, version_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (document_id, name, num_pages, num_chunks, file_type, uploaded_at, version_id),
            )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO parents
                    (parent_id, document_id, document_name, page_number, chunk_index,
                     header_context, parent_text, summary, version_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        parent["parent_id"],
                        document_id,
                        name,
                        parent["page_number"],
                        parent["chunk_index"],
                        parent["header_context"],
                        parent["parent_text"],
                        parent["summary"],
                        version_id,
                    )
                    for parent in parents
                ],
            )
            self.conn.executemany(
                """
                INSERT INTO documents_fts
                    (document_id, chunk_id, document_name, page_number, header_context, parent_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                fts_rows,
            )
            self.conn.commit()

    def fetch_parent(self, parent_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM parents WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
        return dict(row) if row else None

    def keyword_search(
        self,
        fts_query: str,
        limit: int,
        document_id: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT document_id, chunk_id, document_name, page_number, header_context, parent_text,
                   bm25(documents_fts) AS score
            FROM documents_fts
            WHERE documents_fts MATCH ?
        """
        params: List[Any] = [fts_query]
        if document_id:
            sql += " AND document_id = ?"
            params.append(document_id)
        blocked = [item for item in (exclude_ids or []) if item]
        if blocked:
            sql += f" AND document_id NOT IN ({', '.join('?' for _ in blocked)})"
            params.extend(blocked)
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(limit)
        with self._lock:
            try:
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT document_id, document_name, COUNT(*) AS chunk_count, MAX(page_number) AS max_page
                FROM documents_fts
                GROUP BY document_id, document_name
                """
            ).fetchall()
            deleted = {
                row["document_id"]
                for row in self.conn.execute(
                    "SELECT document_id FROM catalog WHERE status = 'deleted'"
                ).fetchall()
            }
        documents = []
        for row in rows:
            if row["document_id"] in deleted:
                continue
            documents.append(
                {
                    "id": row["document_id"],
                    "name": row["document_name"],
                    "chunk_count": row["chunk_count"],
                    "max_page": row["max_page"] or 1,
                }
            )
        return documents

    def document_exists(self, document_id: str) -> bool:
        with self._lock:
            fts = self.conn.execute(
                "SELECT 1 FROM documents_fts WHERE document_id = ? LIMIT 1",
                (document_id,),
            ).fetchone()
            catalog = self.conn.execute(
                "SELECT status FROM catalog WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        if catalog and catalog["status"] == "deleted":
            return False
        return fts is not None or catalog is not None

    def mark_deleted(self, document_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM documents_fts WHERE document_id = ?", (document_id,))
            self.conn.execute("DELETE FROM parents WHERE document_id = ?", (document_id,))
            self.conn.execute(
                """
                INSERT INTO catalog (document_id, name, num_pages, num_chunks, file_type, uploaded_at, version_id, status)
                VALUES (?, '', 0, 0, '', ?, '', 'deleted')
                ON CONFLICT(document_id) DO UPDATE SET status = 'deleted'
                """,
                (document_id, _now()),
            )
            self.conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
