"""Workspace records that should survive a restart: settings, threads, jobs, and usage."""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


DEFAULTS = {
    "workspace_name": "Needle",
    "profile_name": "Operator",
    "default_collection": "General",
    "show_traces": "true",
    "allow_downloads": "true",
    "answer_length": "Balanced",
    "citation_style": "Inline numbered",
    "require_citations": "true",
    "withhold_ungrounded": "true",
    "top_k": "30",
    "similarity_threshold": "0.30",
    "max_parents": "5",
    "rrf_k": "60",
    "contextual_embeddings": "true",
    "chunking": "Parent-child",
}


class WorkspaceStore:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                document_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                validation_json TEXT,
                trace_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                message_id TEXT PRIMARY KEY,
                rating TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                query TEXT NOT NULL,
                grounded INTEGER NOT NULL,
                withheld INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                source_count INTEGER NOT NULL,
                best_similarity REAL,
                candidate_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                name TEXT NOT NULL,
                detail TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_policy (
                document_id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                included INTEGER NOT NULL DEFAULT 1,
                citation_required INTEGER NOT NULL DEFAULT 1,
                bytes INTEGER NOT NULL DEFAULT 0,
                stored_name TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                document_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column("messages", "original_query", "TEXT")
        self._ensure_column("messages", "rewritten_query", "TEXT")
        self._ensure_column("document_policy", "content_hash", "TEXT")
        self._ensure_column("document_policy", "deleted", "INTEGER NOT NULL DEFAULT 0")
        # Conversations belong to a visitor id; the signed-in owner (and every non-demo install) is 'owner'.
        self._ensure_column("conversations", "owner", "TEXT NOT NULL DEFAULT 'owner'")
        self._ensure_column("events", "outcome", "TEXT")
        self._ensure_column("events", "retrieval_ms", "INTEGER")
        self._ensure_column("events", "relevance", "REAL")
        self._ensure_column("events", "cited", "INTEGER")
        self._ensure_column("events", "reject_category", "TEXT")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_candidates (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        for key, value in DEFAULTS.items():
            self.conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
        # Retired features: members, region, and timezone did nothing server-side.
        self.conn.execute("DELETE FROM settings WHERE key IN ('region', 'timezone')")
        self.conn.execute("DROP TABLE IF EXISTS members")
        self.conn.execute(
            "UPDATE jobs SET status = 'failed', error = 'The server restarted before this upload finished.', updated_at = ? WHERE status IN ('queued', 'processing')",
            (_now(),),
        )
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        present = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def settings(self) -> Dict[str, Any]:
        with self._lock:
            rows = self.conn.execute("SELECT key, value FROM settings").fetchall()
        raw = {row["key"]: row["value"] for row in rows}
        return {
            "workspace_name": raw.get("workspace_name", DEFAULTS["workspace_name"]),
            "profile_name": raw.get("profile_name", DEFAULTS["profile_name"]),
            "default_collection": raw.get("default_collection", DEFAULTS["default_collection"]),
            "show_traces": raw.get("show_traces", "true") == "true",
            "allow_downloads": raw.get("allow_downloads", "true") == "true",
            "answer_length": raw.get("answer_length", "Balanced"),
            "citation_style": raw.get("citation_style", "Inline numbered"),
            "require_citations": raw.get("require_citations", "true") == "true",
            "withhold_ungrounded": raw.get("withhold_ungrounded", "true") == "true",
            "top_k": _bounded_int(raw.get("top_k"), 30, 1, 100),
            "similarity_threshold": _bounded_float(raw.get("similarity_threshold"), 0.30, 0, 1),
            "max_parents": _bounded_int(raw.get("max_parents"), 5, 1, 12),
            "rrf_k": _bounded_int(raw.get("rrf_k"), 60, 1, 200),
            "contextual_embeddings": raw.get("contextual_embeddings", "true") == "true",
            "chunking": raw.get("chunking", "Parent-child"),
        }

    def save_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = self.settings()
        mapping = {
            "workspace_name": str(payload.get("workspace_name", current["workspace_name"])).strip() or current["workspace_name"],
            "profile_name": str(payload.get("profile_name", current["profile_name"])).strip() or current["profile_name"],
            "default_collection": str(payload.get("default_collection", current["default_collection"])).strip() or "General",
            "show_traces": "true" if payload.get("show_traces", current["show_traces"]) else "false",
            "allow_downloads": "true" if payload.get("allow_downloads", current["allow_downloads"]) else "false",
            "answer_length": str(payload.get("answer_length", current["answer_length"])),
            "citation_style": str(payload.get("citation_style", current["citation_style"])),
            "require_citations": "true" if payload.get("require_citations", current["require_citations"]) else "false",
            "withhold_ungrounded": "true" if payload.get("withhold_ungrounded", current["withhold_ungrounded"]) else "false",
            "top_k": str(_bounded_int(payload.get("top_k"), current["top_k"], 1, 100)),
            "similarity_threshold": f"{_bounded_float(payload.get('similarity_threshold'), current['similarity_threshold'], 0, 1):.2f}",
            "max_parents": str(_bounded_int(payload.get("max_parents"), current["max_parents"], 1, 12)),
            "rrf_k": str(_bounded_int(payload.get("rrf_k"), current["rrf_k"], 1, 200)),
            "contextual_embeddings": "true" if payload.get("contextual_embeddings", current["contextual_embeddings"]) else "false",
            "chunking": payload.get("chunking") or current["chunking"],
        }
        with self._lock:
            self.conn.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                list(mapping.items()),
            )
            self.conn.commit()
        return self.settings()

    def list_conversations(self, query: str = "", owner: str = "owner") -> List[Dict[str, Any]]:
        sql = """
            SELECT c.id, c.title, c.document_id, c.created_at, c.updated_at, (
                SELECT COUNT(*) FROM messages m
                WHERE m.conversation_id = c.id AND m.role = 'assistant' AND m.sources_json IS NOT NULL AND m.sources_json != '[]'
            ) AS source_threads
            FROM conversations c
            WHERE c.owner = ?
        """
        params: List[Any] = [owner]
        if query.strip():
            sql += " AND c.title LIKE ? ESCAPE '\\'"
            escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        sql += " ORDER BY c.updated_at DESC LIMIT 200"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def ensure_conversation(
        self, conversation_id: Optional[str], title: str, document_id: Optional[str], owner: str = "owner"
    ) -> str:
        now = _now()
        with self._lock:
            if conversation_id:
                # Someone else's conversation id is treated as unknown, never as a hint that it exists.
                row = self.conn.execute(
                    "SELECT id FROM conversations WHERE id = ? AND owner = ?", (conversation_id, owner)
                ).fetchone()
                if row:
                    self.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
                    self.conn.commit()
                    return conversation_id
            new_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO conversations (id, title, document_id, created_at, updated_at, owner) VALUES (?, ?, ?, ?, ?, ?)",
                (new_id, title.strip()[:120] or "New conversation", document_id, now, now, owner),
            )
            self.conn.commit()
            return new_id

    def conversation_exists(self, conversation_id: str, owner: str = "owner") -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND owner = ?", (conversation_id, owner)
            ).fetchone()
        return row is not None

    def create_conversation(self, owner: str = "owner") -> Dict[str, Any]:
        now = _now()
        new_id = str(uuid.uuid4())
        with self._lock:
            self.conn.execute(
                "INSERT INTO conversations (id, title, document_id, created_at, updated_at, owner) VALUES (?, ?, NULL, ?, ?, ?)",
                (new_id, "New conversation", now, now, owner),
            )
            self.conn.commit()
        return {"id": new_id, "title": "New conversation", "document_id": None, "created_at": now, "updated_at": now}

    def messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT m.*, f.rating FROM messages m
                LEFT JOIN feedback f ON f.message_id = m.id
                WHERE m.conversation_id = ?
                ORDER BY m.created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["sources"] = _loads(item.pop("sources_json"))
            item["validation"] = _loads(item.pop("validation_json"))
            item["trace"] = _loads(item.pop("trace_json"))
            result.append(item)
        return result

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[list] = None,
        validation: Optional[dict] = None,
        trace: Optional[dict] = None,
        original_query: Optional[str] = None,
        rewritten_query: Optional[str] = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        now = _now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, sources_json, validation_json, trace_json,
                    created_at, original_query, rewritten_query
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    json.dumps(sources or []),
                    json.dumps(validation) if validation else None,
                    json.dumps(trace) if trace else None,
                    now,
                    original_query,
                    rewritten_query,
                ),
            )
            if role == "user":
                self.conn.execute(
                    "UPDATE conversations SET title = CASE WHEN title = 'New conversation' THEN ? ELSE title END, updated_at = ? WHERE id = ?",
                    (content.strip()[:120], now, conversation_id),
                )
            else:
                self.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            self.conn.commit()
        return message_id

    def set_rewritten_query(self, message_id: str, rewritten: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE messages SET rewritten_query = ? WHERE id = ?", (rewritten[:500], message_id))
            self.conn.commit()

    def set_feedback(self, message_id: str, rating: str, owner: str = "owner") -> bool:
        if rating not in {"helpful", "unhelpful"}:
            return False
        with self._lock:
            row = self.conn.execute(
                """
                SELECT m.id FROM messages m JOIN conversations c ON c.id = m.conversation_id
                WHERE m.id = ? AND m.role = 'assistant' AND c.owner = ?
                """,
                (message_id, owner),
            ).fetchone()
            if not row:
                return False
            self.conn.execute(
                "INSERT INTO feedback (message_id, rating, created_at) VALUES (?, ?, ?) ON CONFLICT(message_id) DO UPDATE SET rating = excluded.rating, created_at = excluded.created_at",
                (message_id, rating, _now()),
            )
            if rating == "unhelpful":
                question = self.conn.execute(
                    """
                    SELECT content FROM messages
                    WHERE conversation_id = (SELECT conversation_id FROM messages WHERE id = ?)
                      AND role = 'user' AND created_at <= (SELECT created_at FROM messages WHERE id = ?)
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (message_id, message_id),
                ).fetchone()
                if question:
                    self.conn.execute(
                        "INSERT INTO eval_candidates (id, message_id, query, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
                        (str(uuid.uuid4()), message_id, question["content"], _now()),
                    )
            self.conn.commit()
        return True

    def record_event(self, **fields: Any) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO events (
                    id, created_at, query, grounded, withheld, latency_ms, source_count, best_similarity,
                    candidate_count, outcome, retrieval_ms, relevance, cited, reject_category
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    _now(),
                    fields.get("query", "")[:500],
                    1 if fields.get("grounded") else 0,
                    1 if fields.get("withheld") else 0,
                    int(fields.get("latency_ms") or 0),
                    int(fields.get("source_count") or 0),
                    fields.get("best_similarity"),
                    int(fields.get("candidate_count") or 0),
                    fields.get("outcome") or "",
                    int(fields.get("retrieval_ms") or 0),
                    fields.get("relevance"),
                    1 if fields.get("cited") else 0,
                    fields.get("reject_category"),
                ),
            )
            self.conn.commit()

    def questions_today(self) -> int:
        """Questions asked since 00:00 UTC; the public demo's daily cap counts these."""
        start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM events WHERE created_at >= ?", (start,)).fetchone()
        return int(row["n"])

    def record_run(self, name: str, detail: str, status: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO runs (id, created_at, name, detail, status) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), _now(), name, detail, status),
            )
            self.conn.commit()

    def recent_runs(self, limit: int = 12) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _window(self, since: str, until: Optional[str] = None) -> Dict[str, Any]:
        sql = "SELECT * FROM events WHERE created_at >= ?"
        params: List[Any] = [since]
        if until:
            sql += " AND created_at < ?"
            params.append(until)
        events = self.conn.execute(sql + " ORDER BY created_at ASC", params).fetchall()
        feedback_sql = "SELECT rating FROM feedback WHERE created_at >= ?" + (" AND created_at < ?" if until else "")
        feedback = self.conn.execute(feedback_sql, params).fetchall()
        total = len(events)
        answered = [event for event in events if not event["withheld"]]
        relevance = [event["relevance"] for event in events if event["relevance"] is not None]
        retrieval = sorted(event["retrieval_ms"] for event in events if event["retrieval_ms"])
        latencies = sorted(event["latency_ms"] for event in events if event["latency_ms"])
        helpful = sum(1 for row in feedback if row["rating"] == "helpful")
        return {
            "events": events,
            "questions": total,
            "grounded_rate": _rate(sum(1 for event in events if event["grounded"]), total),
            "withheld_rate": _rate(sum(1 for event in events if event["withheld"]), total),
            "helpful_rate": _rate(helpful, len(feedback)),
            "ratings": len(feedback),
            "citation_coverage": _rate(sum(1 for event in answered if event["cited"]), len(answered)),
            "mean_relevance": round(sum(relevance) / len(relevance), 4) if relevance else None,
            "retrieval_p50_ms": _p50(retrieval),
            "latency_p50_ms": _p50(latencies),
        }

    def analytics(self, days: int) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        since = (now - timedelta(days=days)).isoformat()
        prior_since = (now - timedelta(days=days * 2)).isoformat()
        with self._lock:
            current = self._window(since)
            prior = self._window(prior_since, since)

            def gap_rows(outcome: str):
                return self.conn.execute(
                    """
                    SELECT MIN(query) AS query, COUNT(*) AS attempts, MAX(best_similarity) AS best_similarity,
                           MAX(created_at) AS last_seen, ? AS outcome
                    FROM events
                    WHERE created_at >= ? AND outcome = ?
                    GROUP BY lower(trim(query))
                    ORDER BY attempts DESC, last_seen DESC
                    LIMIT 8
                    """,
                    (outcome, since, outcome),
                ).fetchall()

            no_coverage = gap_rows("no_coverage")
            check_failed = gap_rows("check_failed")
        buckets: Dict[str, Dict[str, int]] = {}
        for offset in range(days - 1, -1, -1):
            day = (now - timedelta(days=offset)).date().isoformat()
            buckets[day] = {"questions": 0, "grounded": 0}
        for event in current["events"]:
            slot = buckets.setdefault(event["created_at"][:10], {"questions": 0, "grounded": 0})
            slot["questions"] += 1
            slot["grounded"] += 1 if event["grounded"] else 0
        metrics = {key: value for key, value in current.items() if key != "events"}
        previous = {key: value for key, value in prior.items() if key != "events"}
        return {
            "days": days,
            **metrics,
            "prior": previous,
            "series": [{"day": day, **counts} for day, counts in sorted(buckets.items())],
            "gaps": [
                {
                    "query": row["query"],
                    "attempts": row["attempts"],
                    "best_similarity": row["best_similarity"],
                    "last_seen": row["last_seen"],
                    "outcome": row["outcome"],
                }
                for row in list(no_coverage) + list(check_failed)
            ],
            "gaps_no_coverage": [row["query"] for row in no_coverage],
            "gaps_check_failed": [row["query"] for row in check_failed],
        }

    def pipeline_stats(self, days: int = 30) -> Dict[str, Any]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            runs = self.conn.execute("SELECT status FROM runs WHERE created_at >= ?", (since,)).fetchall()
            retrieval = [
                row["retrieval_ms"]
                for row in self.conn.execute(
                    "SELECT retrieval_ms FROM events WHERE created_at >= ? AND retrieval_ms > 0", (since,)
                ).fetchall()
            ]
            handoff = self.conn.execute(
                "SELECT created_at, detail FROM runs WHERE name = 'Version handoff' AND status = 'Success' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        finished = [row for row in runs if row["status"] in {"Success", "Failed"}]
        return {
            "run_success_rate": _rate(sum(1 for row in finished if row["status"] == "Success"), len(finished)),
            "runs_counted": len(finished),
            "mean_retrieval_ms": round(sum(retrieval) / len(retrieval)) if retrieval else None,
            "last_handoff_at": handoff["created_at"] if handoff else None,
            "last_handoff_detail": handoff["detail"] if handoff else None,
        }

    def start_job(self, filename: str) -> str:
        job_id = str(uuid.uuid4())
        now = _now()
        with self._lock:
            self.conn.execute(
                "INSERT INTO jobs (id, filename, status, error, document_id, created_at, updated_at) VALUES (?, ?, 'queued', NULL, NULL, ?, ?)",
                (job_id, filename, now, now),
            )
            self.conn.commit()
        return job_id

    def update_job(self, job_id: str, status: str, error: Optional[str] = None, document_id: Optional[str] = None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET status = ?, error = ?, document_id = COALESCE(?, document_id), updated_at = ? WHERE id = ?",
                (status, error, document_id, _now(), job_id),
            )
            self.conn.commit()

    def job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def active_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'processing') ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def find_hash(self, content_hash: str) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT document_id FROM document_policy WHERE content_hash = ? AND deleted = 0",
                (content_hash,),
            ).fetchone()
        return row["document_id"] if row else None

    def tombstone(self, document_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE document_policy SET deleted = 1, included = 0 WHERE document_id = ?",
                (document_id,),
            )
            self.conn.commit()

    def set_policy(self, document_id: str, *, collection: str, byte_size: int, stored_name: str, content_hash: str = "") -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO document_policy (document_id, collection, included, citation_required, bytes, stored_name, content_hash, deleted)
                VALUES (?, ?, 1, 1, ?, ?, ?, 0)
                ON CONFLICT(document_id) DO UPDATE SET
                    bytes = excluded.bytes,
                    stored_name = excluded.stored_name,
                    content_hash = excluded.content_hash,
                    deleted = 0,
                    included = 1
                """,
                (document_id, collection, byte_size, stored_name, content_hash),
            )
            self.conn.commit()

    def update_policy(self, document_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = self.policy(document_id)
        if not current:
            return None
        included = fields.get("included", current["included"])
        citation_required = fields.get("citation_required", current["citation_required"])
        collection = str(fields.get("collection", current["collection"])).strip() or current["collection"]
        with self._lock:
            self.conn.execute(
                "UPDATE document_policy SET included = ?, citation_required = ?, collection = ? WHERE document_id = ?",
                (1 if included else 0, 1 if citation_required else 0, collection, document_id),
            )
            self.conn.commit()
        return self.policy(document_id)

    def policy(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM document_policy WHERE document_id = ?", (document_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["included"] = bool(item["included"])
        item["citation_required"] = bool(item["citation_required"])
        item["deleted"] = bool(item.get("deleted"))
        return item

    def policies(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM document_policy").fetchall()
        result = {}
        for row in rows:
            item = dict(row)
            item["included"] = bool(item["included"])
            item["citation_required"] = bool(item["citation_required"])
            item["deleted"] = bool(item.get("deleted"))
            result[item["document_id"]] = item
        return result

    def citation_required_ids(self) -> List[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT document_id FROM document_policy WHERE citation_required = 1 AND deleted = 0"
            ).fetchall()
        return [row["document_id"] for row in rows]

    def excluded_document_ids(self) -> List[str]:
        with self._lock:
            rows = self.conn.execute("SELECT document_id FROM document_policy WHERE included = 0").fetchall()
        return [row["document_id"] for row in rows]

    def forget_document(self, document_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM document_policy WHERE document_id = ?", (document_id,))
            self.conn.commit()

    def reset(self) -> None:
        with self._lock:
            for table in (
                "conversations",
                "messages",
                "feedback",
                "events",
                "runs",
                "document_policy",
                "jobs",
                "eval_candidates",
            ):
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute("DELETE FROM settings")
            for key, value in DEFAULTS.items():
                self.conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (key, value))
            self.conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: Optional[str]):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _p50(values: List[int]) -> Optional[int]:
    return values[len(values) // 2] if values else None


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
