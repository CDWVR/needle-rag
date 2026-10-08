"""HTTP-level security and behaviour tests. Needs the real dependencies (FastAPI, Chroma, the
local embedder) but no network: OpenRouter is disabled, so chat is never exercised here.

    python -m unittest discover -s backend/tests -t backend
"""

import io
import os
import sys
import tempfile
import time
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = tempfile.mkdtemp(prefix="needle-api-test-")
TOKEN = "test-access-token-0123456789"
os.environ.update({
    "NEEDLE_DATA_DIR": DATA_DIR,
    "NEEDLE_ACCESS_TOKEN": TOKEN,
    "NEEDLE_SESSION_SECRET": "test-session-secret-0123456789",
    "NEEDLE_ALLOWED_HOSTS": "testserver",
    "NEEDLE_MAX_UPLOAD_MB": "1",
    "OPENROUTER_API_KEY": "",
})

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

BEARER = {"Authorization": f"Bearer {TOKEN}"}
WRITE = {"X-Needle-CSRF": "1"}
TEXT = ("Ottermere ships the Tern-3 cart. Its rated payload is 120 kg and it runs for 9.5 hours. " * 4).encode()


def office_file(payload: bytes = b"<x/>", extra: int = 0) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", payload + b" " * extra)
    return buffer.getvalue()


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_health_is_public_and_says_nothing(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_api_requires_a_session(self):
        self.assertEqual(self.client.get("/api/settings").status_code, 401)
        self.assertEqual(self.client.get("/api/documents").status_code, 401)
        self.assertEqual(self.client.get("/api/settings", headers={"Authorization": "Bearer wrong-token-value"}).status_code, 401)
        self.assertEqual(self.client.get("/api/settings", headers=BEARER).status_code, 200)

    def test_cookie_session_with_csrf(self):
        self.assertEqual(self.client.post("/api/auth/login", json={"token": "nope"}).status_code, 401)
        login = self.client.post("/api/auth/login", json={"token": TOKEN})
        self.assertEqual(login.status_code, 200)
        cookie = login.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        self.assertEqual(self.client.get("/api/settings").status_code, 200)
        # Writes with the cookie need the CSRF header and a same-origin Origin.
        self.assertEqual(self.client.post("/api/conversations").status_code, 403)
        self.assertEqual(self.client.post("/api/conversations", headers=WRITE).status_code, 200)
        cross = {**WRITE, "Origin": "https://evil.example"}
        self.assertEqual(self.client.post("/api/conversations", headers=cross).status_code, 403)
        self.client.post("/api/auth/logout", headers=WRITE)
        self.assertEqual(self.client.get("/api/settings").status_code, 401)

    def test_tampered_session_is_rejected(self):
        self.client.post("/api/auth/login", json={"token": TOKEN})
        value = self.client.cookies.get("needle_session")
        self.client.cookies.set("needle_session", value[:-2] + ("00" if value[-2:] != "00" else "11"))
        self.assertEqual(self.client.get("/api/settings").status_code, 401)

    def test_login_is_rate_limited(self):
        client = TestClient(main.app)
        statuses = [client.post("/api/auth/login", json={"token": "guess"}).status_code for _ in range(12)]
        self.assertIn(429, statuses)
        main.security.limiter._hits.clear()

    def test_unknown_hosts_are_refused(self):
        self.assertEqual(self.client.get("/health", headers={"Host": "evil.example"}).status_code, 400)

    def test_api_schema_is_not_published(self):
        self.assertEqual(self.client.get("/api/openapi.json", headers=BEARER).status_code, 404)
        self.assertEqual(self.client.get("/api/docs", headers=BEARER).status_code, 404)
        # Unknown paths are refused before routing when there is no session.
        self.assertEqual(self.client.get("/docs").status_code, 401)


class ReadEndpointTests(unittest.TestCase):
    """Every read the UI makes on load must work with the query strings the UI sends."""

    def test_ui_reads(self):
        client = TestClient(main.app)
        for path in ("/api/settings", "/api/documents", "/api/index", "/api/index/versions", "/api/eval/latest",
                     "/api/conversations", "/api/conversations?q=100%25_", "/api/analytics?days=7",
                     "/api/analytics?days=30", "/api/analytics?days=90", "/api/analytics/export?days=30"):
            self.assertEqual(client.get(path, headers=BEARER).status_code, 200, path)
        self.assertEqual(client.get("/api/analytics?days=5", headers=BEARER).status_code, 400)


class HeaderTests(unittest.TestCase):
    def test_security_headers_on_pages_and_errors(self):
        client = TestClient(main.app)
        for response in (client.get("/"), client.get("/api/settings")):
            headers = response.headers
            self.assertIn("script-src 'self'", headers["content-security-policy"])
            self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])
            self.assertEqual(headers["x-content-type-options"], "nosniff")
            self.assertEqual(headers["x-frame-options"], "DENY")
            self.assertEqual(headers["referrer-policy"], "no-referrer")
        self.assertEqual(client.get("/api/settings", headers=BEARER).headers["cache-control"], "no-store")


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        main.security.limiter._hits.clear()

    def upload(self, name, data):
        return self.client.post("/api/upload", headers=BEARER, files={"file": (name, data)})

    def test_extension_and_content_must_agree(self):
        self.assertEqual(self.upload("run.exe", b"MZ...").status_code, 400)
        self.assertEqual(self.upload("fake.pdf", b"not a pdf at all").status_code, 400)
        self.assertEqual(self.upload("fake.docx", b"plain text pretending").status_code, 400)
        self.assertEqual(self.upload("binary.txt", b"abc\x00def").status_code, 400)
        self.assertEqual(self.upload("latin.txt", "caf\xe9".encode("latin-1")).status_code, 400)
        self.assertEqual(self.upload("empty.txt", b"").status_code, 400)

    def test_zip_bombs_are_rejected(self):
        bomb = office_file(extra=900_000)  # ~0.9 MB of spaces compresses to a few KB
        self.assertEqual(self.upload("bomb.docx", bomb).status_code, 400)

    def test_size_limit(self):
        self.assertEqual(self.upload("big.txt", b"a" * (1024 * 1024 + 10)).status_code, 413)

    def test_upload_is_queued_indexed_and_downloaded_safely(self):
        queued = self.upload("../../secrets/notes.txt", TEXT)
        self.assertEqual(queued.status_code, 202)
        body = queued.json()
        self.assertEqual(body["name"], "notes.txt")  # directories are stripped
        deadline = time.time() + 120
        job = {}
        while time.time() < deadline:
            job = self.client.get(f"/api/jobs/{body['job_id']}", headers=BEARER).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.5)
        self.assertEqual(job["status"], "completed", job)
        duplicate = self.upload("again.txt", TEXT).json()
        self.assertEqual(duplicate["status"], "duplicate")
        download = self.client.get(f"/api/documents/{job['document_id']}/file", headers=BEARER)
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.headers["content-disposition"].startswith("attachment"))
        self.assertTrue(download.headers["content-type"].startswith("text/plain"))
        self.assertEqual(download.content, TEXT)

    def test_ids_are_validated(self):
        self.assertEqual(self.client.get("/api/documents/not-a-uuid", headers=BEARER).status_code, 422)
        self.assertEqual(self.client.get("/api/jobs/1;drop", headers=BEARER).status_code, 422)


if __name__ == "__main__":
    unittest.main()
