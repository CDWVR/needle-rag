"""Authentication, request hardening, rate limits, and upload validation.

Threat model: Needle holds private documents and spends API credit, so every API
call must come from someone holding the workspace access token. It is a single-
tenant app: one token, many browser sessions. See SECURITY.md.
"""

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import threading
import time
import zipfile
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from config import env_bool, env_int, env_str
from paths import data_path

log = logging.getLogger("needle.security")

SESSION_COOKIE = "needle_session"
CSRF_HEADER = "x-needle-csrf"
SESSION_TTL_SECONDS = env_int("NEEDLE_SESSION_HOURS", 12) * 3600
AUTH_DISABLED = env_bool("NEEDLE_AUTH_DISABLED", False)
# Public demo: anyone may read the sample corpus and ask questions; only the signed-in owner may change anything.
DEMO_MODE = env_bool("NEEDLE_DEMO_MODE", False)
DEMO_DAILY_QUESTIONS = env_int("NEEDLE_DEMO_DAILY_QUESTIONS", 300)
DEMO_MAX_QUESTION_CHARS = env_int("NEEDLE_DEMO_MAX_QUESTION_CHARS", 500)
VISITOR_COOKIE = "needle_visitor"
COOKIE_SECURE = env_bool("NEEDLE_COOKIE_SECURE", False)
# Paths reachable without a session: the page shell, static assets, health, and sign-in itself.
PUBLIC_PATHS = {"/", "/health", "/api/auth/login", "/api/auth/logout", "/api/auth/session"}
# What a demo visitor may call. Everything else needs the owner session.
VISITOR_ROUTES = {
    ("GET", "/api/settings"), ("GET", "/api/conversations"), ("POST", "/api/conversations"),
    ("GET", "/api/documents"), ("GET", "/api/index"), ("GET", "/api/index/versions"), ("GET", "/api/eval/latest"),
    ("GET", "/api/analytics"), ("POST", "/api/chat"), ("POST", "/api/transcribe"),
}
VISITOR_PREFIXES = (
    ("GET", "/api/conversations/"), ("PATCH", "/api/conversations/"), ("DELETE", "/api/conversations/"),  # own threads only
    ("GET", "/api/documents/"), ("POST", "/api/messages/"),
)
PUBLIC_PREFIXES = ("/static/",)
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


# --- secrets -----------------------------------------------------------------------

def _secret_file(name: str) -> str:
    """Read a secret from data/secrets/<name>, creating it with a random value on first use."""
    directory = data_path("secrets")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
        if value:
            return value
        os.remove(path)  # an empty secret is no secret
    value = secrets.token_urlsafe(32)
    # O_EXCL: never overwrite a secret another process just wrote.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value)
    if name == "access_token":
        # Shown once, on the machine running the server, so the owner can sign in.
        print(f"\nNeedle generated an access token (also saved to {path}):\n\n    {value}\n", flush=True)
    return value


_ACCESS_TOKEN = env_str("NEEDLE_ACCESS_TOKEN", "") or _secret_file("access_token")
_SESSION_KEY = (env_str("NEEDLE_SESSION_SECRET", "") or _secret_file("session_secret")).encode()
if len(_ACCESS_TOKEN) < 16:
    raise RuntimeError("NEEDLE_ACCESS_TOKEN must be at least 16 characters.")


def token_matches(candidate: str) -> bool:
    return hmac.compare_digest((candidate or "").encode(), _ACCESS_TOKEN.encode())


# --- sessions ------------------------------------------------------------------------

def _sign(payload: bytes) -> str:
    return hmac.new(_SESSION_KEY, payload, hashlib.sha256).hexdigest()


def issue_session() -> str:
    payload = json.dumps({"exp": int(time.time()) + SESSION_TTL_SECONDS, "n": secrets.token_hex(8)}).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{encoded}.{_sign(payload)}"


def session_valid(cookie: Optional[str]) -> bool:
    if not cookie or "." not in cookie:
        return False
    encoded, signature = cookie.rsplit(".", 1)
    try:
        payload = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, TypeError):
        return False
    if not hmac.compare_digest(_sign(payload), signature):
        return False
    try:
        return int(json.loads(payload)["exp"]) > time.time()
    except (ValueError, KeyError, TypeError):
        return False


def set_session_cookie(response: Response, request: Request) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        issue_session(),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="strict",
        secure=COOKIE_SECURE or request.url.scheme == "https",
        path="/",
    )


def is_authenticated(request: Request) -> bool:
    """True for the owner: a valid token or session (or auth switched off for local development)."""
    if AUTH_DISABLED:
        return True
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer ") and token_matches(header[7:].strip()):
        return True
    return session_valid(request.cookies.get(SESSION_COOKIE))


def role_of(request: Request) -> str:
    return "owner" if is_authenticated(request) else ("visitor" if DEMO_MODE else "anonymous")


def visitor_allowed(method: str, path: str) -> bool:
    if (method, path) in VISITOR_ROUTES:
        return True
    return any(method == m and path.startswith(prefix) and not path.endswith("/file") for m, prefix in VISITOR_PREFIXES)


def conversation_owner(request: Request) -> str:
    """Whose conversations this request may see: the owner's, or the visitor's own."""
    if getattr(request.state, "role", "owner") == "owner":
        return "owner"
    return "visitor:" + getattr(request.state, "visitor_id", "none")


def _clean_visitor_id(value: Optional[str]) -> Optional[str]:
    return value if value and re.fullmatch(r"[A-Za-z0-9_-]{20,64}", value) else None


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True  # same-origin fetches from older browsers and non-browser clients omit it
    host = request.headers.get("host", "")
    return origin.split("://", 1)[-1] == host


class AuthMiddleware(BaseHTTPMiddleware):
    """Every /api call needs a session or the bearer token; cookie-authenticated writes also
    need the CSRF header (which a cross-site page cannot set) and a same-origin Origin."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        request.state.role = role_of(request)
        new_visitor = None
        if DEMO_MODE:
            visitor = _clean_visitor_id(request.cookies.get(VISITOR_COOKIE))
            if not visitor:
                visitor = new_visitor = secrets.token_urlsafe(24)
            request.state.visitor_id = visitor
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return self._with_visitor_cookie(await call_next(request), request, new_visitor)
        if request.state.role == "anonymous":
            return JSONResponse({"detail": "Sign in to continue."}, status_code=401)
        uses_cookie = not request.headers.get("authorization", "").lower().startswith("bearer ")
        if uses_cookie and request.method not in SAFE_METHODS and not AUTH_DISABLED:
            if request.headers.get(CSRF_HEADER) != "1" or not _same_origin(request):
                return JSONResponse({"detail": "Request blocked: missing or cross-site request header."}, status_code=403)
        if request.state.role == "visitor" and not visitor_allowed(request.method, path):
            return JSONResponse({"detail": "This is disabled in the public demo."}, status_code=403)
        return self._with_visitor_cookie(await call_next(request), request, new_visitor)

    @staticmethod
    def _with_visitor_cookie(response: Response, request: Request, new_visitor: Optional[str]) -> Response:
        if new_visitor:
            response.set_cookie(
                VISITOR_COOKIE, new_visitor, max_age=30 * 24 * 3600, httponly=True, samesite="strict",
                secure=COOKIE_SECURE or request.url.scheme == "https", path="/",
            )
        return response


# --- response headers ---------------------------------------------------------------

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    # Inline style attributes size the chart bars; no inline script is allowed.
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: blob:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Permissions-Policy", "camera=(), geolocation=(), microphone=(self), payment=()")
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        if request.url.path.startswith("/api/"):
            headers.setdefault("Cache-Control", "no-store")
        if request.url.scheme == "https":
            headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


# --- rate limits ------------------------------------------------------------------------

RATE_LIMITS = {
    "login": (10, 300),     # 10 attempts per 5 minutes: slows token guessing
    # Across all clients: a backstop for when the per-client key can be spoofed (a proxy that
    # trusts any X-Forwarded-For). The owner signs in rarely; 30 tries in 5 minutes is plenty.
    "login_global": (30, 300),
    "chat": (30, 60),
    "visitor_chat": (6, 60),        # public demo: a few questions a minute...
    "visitor_chat_hour": (40, 3600),  # ...and a bounded number per hour, per client
    "voice": (4, 60),                # speech-to-text costs money per second of audio
    "voice_hour": (20, 3600),
    "upload": (12, 60),
    "refresh": (3, 300),
    "write": (120, 60),
}


class RateLimiter:
    """Sliding window per (bucket, client). Memory stays bounded: idle clients are pruned."""

    def __init__(self, limits: Dict[str, Tuple[int, int]], max_clients: int = 10_000):
        self.limits = limits
        self.max_clients = max_clients
        self._hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, bucket: str, client: str) -> None:
        limit, window = self.limits[bucket]
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(bucket, client)]
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(window - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Wait a moment and try again.",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)
            if len(self._hits) > self.max_clients:
                for key in [key for key, queue in self._hits.items() if not queue or now - queue[-1] > 600]:
                    del self._hits[key]


limiter = RateLimiter(RATE_LIMITS)


def client_id(request: Request) -> str:
    # Behind a proxy, run uvicorn with --proxy-headers so request.client is the real caller.
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, bucket: str) -> None:
    limiter.check(bucket, client_id(request))


def demo_question_gate(request: Request, questions_today: int) -> None:
    """Public-demo limits on spend: tighter per-client rates and a global daily cap."""
    if getattr(request.state, "role", "owner") != "visitor":
        return
    if questions_today >= DEMO_DAILY_QUESTIONS:
        raise HTTPException(status_code=429, detail="The demo has reached today's question limit. Try again tomorrow.")
    limiter.check("visitor_chat", client_id(request))
    limiter.check("visitor_chat_hour", client_id(request))


# --- uploads ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = env_int("NEEDLE_MAX_UPLOAD_MB", 50) * 1024 * 1024
MAX_PDF_PAGES = env_int("NEEDLE_MAX_PDF_PAGES", 2000)
# Office files are zip archives; cap what they may expand to (zip bombs).
MAX_UNZIPPED_BYTES = env_int("NEEDLE_MAX_UNZIPPED_MB", 300) * 1024 * 1024
MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_RATIO = 200
TEXT_EXTENSIONS = {".txt", ".md", ".text", ".csv"}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | OFFICE_EXTENSIONS | {".pdf"}
_UNSAFE_NAME = re.compile(r"[\x00-\x1f\x7f/\\]+")


def clean_filename(raw: str) -> str:
    """Display name only (files are stored under a generated id): no paths or control characters."""
    name = _UNSAFE_NAME.sub(" ", os.path.basename((raw or "").replace("\\", "/")))
    stem, ext = os.path.splitext(name.strip())
    if not ext and stem.startswith(".") and "." not in stem[1:]:
        stem, ext = "", stem  # Python reads ".pdf" as a dotfile with no extension
    stem = stem.strip(" .") or "document"
    return stem[:150] + ext.lower()[:10]


def read_upload(file: UploadFile, declared_length: Optional[str]) -> bytes:
    """Read at most MAX_UPLOAD_BYTES without trusting the client's Content-Length."""
    if declared_length and declared_length.isdigit() and int(declared_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    buffer = io.BytesIO()
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        buffer.write(chunk)
        if buffer.tell() > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return buffer.getvalue()


def validate_content(data: bytes, ext: str) -> None:
    """The bytes must be what the extension claims; reject archives that expand without bound."""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a PDF, Word, text, slides, or spreadsheet file.")
    if ext == ".pdf":
        if not data.lstrip()[:5].startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="That file is not a valid PDF.")
        _check_pdf(data)
    elif ext in OFFICE_EXTENSIONS:
        _check_office_zip(data)
    else:
        if b"\x00" in data[:65536]:
            raise HTTPException(status_code=400, detail="That file looks binary, not text.")
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Text files must be UTF-8 encoded.") from exc


def _check_pdf(data: bytes) -> None:
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF releases only ship the `fitz` name
        import fitz
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="That PDF could not be opened.") from exc
    with document:
        if document.needs_pass:
            raise HTTPException(status_code=400, detail="Password-protected PDFs cannot be indexed.")
        if document.page_count > MAX_PDF_PAGES:
            raise HTTPException(status_code=400, detail=f"PDFs over {MAX_PDF_PAGES} pages are not accepted.")


def _check_office_zip(data: bytes) -> None:
    if not data.startswith(b"PK\x03\x04"):
        raise HTTPException(status_code=400, detail="That Office file is not a valid document.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if "[Content_Types].xml" not in archive.namelist():
                raise HTTPException(status_code=400, detail="That Office file is not a valid document.")
            unzipped = sum(entry.file_size for entry in entries)
            packed = sum(entry.compress_size for entry in entries) or 1
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="That Office file is damaged.") from exc
    if len(entries) > MAX_ZIP_ENTRIES or unzipped > MAX_UNZIPPED_BYTES or unzipped / packed > MAX_ZIP_RATIO:
        raise HTTPException(status_code=400, detail="That file expands to an unsafe size and was rejected.")


# Download media types: never let a stored file be served as something a browser executes.
DOWNLOAD_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".text": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
