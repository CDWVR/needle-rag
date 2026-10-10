"""Bug reports from the in-app reporter: what went wrong, where, and a screenshot of the page.

Anyone who can open the workspace (the owner, or a visitor in the public demo) can send one; only
the owner can read them. Reports live in their own SQLite file and folder beside the workspace data,
so deleting workspace data does not delete them. Optionally each new report is announced to a
webhook (Slack, Discord, or anything that accepts a JSON POST), so it reaches the owner directly.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("needle")

MAX_DESCRIPTION = 2000
MAX_SCREENSHOT_BYTES = 4 * 1024 * 1024
KEEP_REPORTS = 300  # the oldest reports (and their screenshots) are removed past this many
STATUSES = ("open", "resolved")


def image_kind(data: bytes) -> Optional[str]:
    """'jpg' or 'png' when the bytes really are a complete JPEG or PNG image, else None."""
    if len(data) > 8 and data[:3] == b"\xff\xd8\xff" and data.rstrip(b"\x00")[-2:] == b"\xff\xd9":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n" and b"IEND" in data[-16:]:
        return "png"
    return None


def clean_line(value: str, limit: int) -> str:
    """One line of plain text: control characters removed, length capped."""
    text = "".join(ch for ch in (value or "") if ch.isprintable())
    return text.strip()[:limit]


def clean_text(value: str, limit: int) -> str:
    """Multi-line plain text: line breaks kept, other control characters removed, length capped."""
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch.isprintable())
    return text.strip()[:limit]


class BugReports:
    def __init__(self, directory: str):
        self.directory = directory
        self.images = os.path.join(directory, "screenshots")
        os.makedirs(self.images, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(os.path.join(directory, "reports.db"), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bug_reports (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                reporter TEXT NOT NULL,
                description TEXT NOT NULL,
                page TEXT NOT NULL,
                viewport TEXT NOT NULL,
                theme TEXT NOT NULL,
                user_agent TEXT NOT NULL,
                screenshot TEXT,
                status TEXT NOT NULL DEFAULT 'open'
            )
            """
        )
        self.conn.commit()

    def add(self, *, role: str, reporter: str, description: str, page: str, viewport: str, theme: str,
            user_agent: str, screenshot: Optional[bytes]) -> Dict[str, Any]:
        report_id = str(uuid.uuid4())
        kind = image_kind(screenshot) if screenshot else None
        if kind:
            with open(self._image_path(report_id, kind), "wb") as handle:
                handle.write(screenshot)
        row = {
            "id": report_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "reporter": reporter[:64],
            "description": clean_text(description, MAX_DESCRIPTION),
            "page": clean_line(page, 200),
            "viewport": clean_line(viewport, 40),
            "theme": clean_line(theme, 10),
            "user_agent": clean_line(user_agent, 300),
            "screenshot": kind,
            "status": "open",
        }
        with self._lock:
            self.conn.execute(
                f"INSERT INTO bug_reports ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", tuple(row.values())
            )
            self.conn.commit()
        self._prune()
        return self._public(row)

    def list(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM bug_reports ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._public(dict(row)) for row in rows]

    def open_count(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM bug_reports WHERE status = 'open'").fetchone()[0])

    def get(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM bug_reports WHERE id = ?", (report_id,)).fetchone()
        return dict(row) if row else None

    def screenshot_path(self, report_id: str) -> Optional[str]:
        row = self.get(report_id)
        if not row or not row.get("screenshot"):
            return None
        path = self._image_path(report_id, row["screenshot"])
        return path if os.path.exists(path) else None

    def set_status(self, report_id: str, status: str) -> bool:
        with self._lock:
            changed = self.conn.execute("UPDATE bug_reports SET status = ? WHERE id = ?", (status, report_id)).rowcount
            self.conn.commit()
        return bool(changed)

    def delete(self, report_id: str) -> bool:
        row = self.get(report_id)
        if not row:
            return False
        self._remove_image(report_id, row.get("screenshot"))
        with self._lock:
            self.conn.execute("DELETE FROM bug_reports WHERE id = ?", (report_id,))
            self.conn.commit()
        return True

    def _prune(self) -> None:
        with self._lock:
            old = self.conn.execute(
                "SELECT id, screenshot FROM bug_reports ORDER BY created_at DESC LIMIT -1 OFFSET ?", (KEEP_REPORTS,)
            ).fetchall()
            for row in old:
                self.conn.execute("DELETE FROM bug_reports WHERE id = ?", (row["id"],))
            self.conn.commit()
        for row in old:
            self._remove_image(row["id"], row["screenshot"])

    def _image_path(self, report_id: str, kind: str) -> str:
        return os.path.join(self.images, f"{report_id}.{kind}")

    def _remove_image(self, report_id: str, kind: Optional[str]) -> None:
        if kind:
            try:
                os.remove(self._image_path(report_id, kind))
            except OSError:
                pass

    @staticmethod
    def _public(row: Dict[str, Any]) -> Dict[str, Any]:
        return {**{key: value for key, value in row.items() if key != "reporter"}, "has_screenshot": bool(row.get("screenshot"))}


def announce(webhook_url: str, report: Dict[str, Any], link: str) -> None:
    """Tell the owner about a new report, off the request thread. Slack reads `text`, Discord reads
    `content`; both ignore the other key. Only the summary is sent, never the screenshot."""
    if not webhook_url.startswith("https://"):
        return
    who = "A demo visitor" if report.get("role") == "visitor" else "The owner"
    summary = report.get("description", "")
    if len(summary) > 280:
        summary = summary[:277] + "..."
    message = f"New Needle bug report. {who} on {report.get('page') or 'the app'}: \"{summary}\" Open it: {link}"
    body = json.dumps({"text": message, "content": message}).encode("utf-8")

    def send() -> None:
        try:
            request = urllib.request.Request(webhook_url, data=body, method="POST",
                                             headers={"Content-Type": "application/json", "User-Agent": "Needle"})
            with urllib.request.urlopen(request, timeout=6):
                pass
        except Exception as exc:  # a missed notification must never fail the report
            log.warning("Bug report webhook failed: %s", exc)

    threading.Thread(target=send, name="bug-webhook", daemon=True).start()
