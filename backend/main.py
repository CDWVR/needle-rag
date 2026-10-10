"""Needle HTTP API and the static workspace UI.

Handlers are plain `def` on purpose: they call SQLite, Chroma, the embedder, and OpenRouter
synchronously, and FastAPI runs `def` handlers in a thread pool so one slow request does not
stall the rest. Every /api route except sign-in requires a session (see security.py).
"""

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

import security
from catalog import (
    delete_document,
    document_passages,
    env_defaults,
    get_all_documents,
    index_status,
    list_index_versions,
    orphaned_documents,
)
from config import CHUNKING_STRATEGIES, env_bool, env_int, env_str
from embedder import embedding_dimensions
from jobs import IngestionQueue
from paths import data_path
from pipeline import PipelineOptions, run_pipeline
import transcribe
from pipeline_logic import InfraError, NeedleError
from versions import PublishBlocked, refresh_active_index, rollback_index
from workspace import WorkspaceStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("needle")

ROOT = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(ROOT, "..", "frontend")
EVAL_REPORTS_DIR = os.path.join(ROOT, "eval", "reports")
UPLOAD_DIR = data_path("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Conversation turns passed to the condense step; older turns add cost, not context.
HISTORY_TURNS = 20
UUID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

workspace = WorkspaceStore(data_path("workspace.db"))
ingestion = IngestionQueue(workspace, UPLOAD_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if security.DEMO_MODE:
        from demo import seed_demo_corpus

        # Background, so the server answers health checks while the sample documents index.
        threading.Thread(target=seed_demo_corpus, args=(workspace,), name="demo-seed", daemon=True).start()
    yield
    ingestion.shutdown(wait=False)


app = FastAPI(
    lifespan=lifespan,
    title="Needle",
    description="Source-grounded knowledge workspace",
    version="2.1.0",
    # The schema lists every route and payload; keep it off unless explicitly wanted.
    docs_url="/api/docs" if env_bool("NEEDLE_ENABLE_API_DOCS", False) else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if env_bool("NEEDLE_ENABLE_API_DOCS", False) else None,
)
# The last middleware added runs first: host check, CORS, security headers (so even a 401 carries
# them), then authentication.
app.add_middleware(security.AuthMiddleware)
app.add_middleware(security.SecurityHeadersMiddleware)
_cors = [origin.strip() for origin in env_str("NEEDLE_CORS_ORIGINS", "").split(",") if origin.strip()]
if _cors:
    # Only needed when another origin hosts the UI; the bundled UI is same-origin.
    app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                       allow_headers=["content-type", security.CSRF_HEADER, "authorization"], allow_credentials=True)
def _allowed_hosts() -> list:
    hosts = [h.strip() for h in env_str("NEEDLE_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if h.strip()]
    # On Railway the public domain is provided, and its health checker calls with its own Host header.
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if railway_domain:
        hosts.append(railway_domain)
    if os.getenv("RAILWAY_ENVIRONMENT", "").strip():
        hosts.append("healthcheck.railway.app")
    return hosts


app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts())
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# --- request bodies ------------------------------------------------------------------

class LoginRequest(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    document_id: Optional[str] = Field(default=None, pattern=UUID)
    conversation_id: Optional[str] = Field(default=None, pattern=UUID)


class SettingsPayload(BaseModel):
    workspace_name: str = Field("Needle", max_length=80)
    profile_name: str = Field("Operator", max_length=80)
    default_collection: str = Field("General", max_length=60)
    show_traces: bool = True
    allow_downloads: bool = True
    answer_length: Literal["Concise", "Balanced", "Detailed"] = "Balanced"
    citation_style: Literal["Inline numbered", "Footnotes", "Source cards"] = "Inline numbered"
    require_citations: bool = True
    withhold_ungrounded: bool = True
    top_k: int = Field(30, ge=1, le=100)
    similarity_threshold: float = Field(0.30, ge=0, le=1)
    max_parents: int = Field(5, ge=1, le=12)
    rrf_k: int = Field(60, ge=1, le=200)
    contextual_embeddings: bool = True
    chunking: Optional[Literal[CHUNKING_STRATEGIES]] = None


class PolicyPayload(BaseModel):
    included: Optional[bool] = None
    citation_required: Optional[bool] = None
    collection: Optional[str] = Field(default=None, max_length=60)


class FeedbackPayload(BaseModel):
    rating: Literal["helpful", "unhelpful", "none"]  # "none" clears a rating (undo)
    reason: Optional[Literal["wrong_source", "incomplete", "should_answer", "other"]] = None


class ConversationPayload(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ResetPayload(BaseModel):
    confirm: str = Field(max_length=16)


# --- helpers -------------------------------------------------------------------------

def _stored_path(policy: Optional[dict]) -> Optional[str]:
    if not policy or not policy.get("stored_name"):
        return None
    path = os.path.join(UPLOAD_DIR, os.path.basename(policy["stored_name"]))
    return path if os.path.exists(path) else None


def _documents():
    policies = workspace.policies()
    documents = []
    for doc in get_all_documents():
        policy = policies.get(doc["id"], {})
        stored = _stored_path(policy)
        documents.append(
            {
                **doc,
                "collection": policy.get("collection") or "General",
                "included": policy.get("included", True),
                "citation_required": policy.get("citation_required", True),
                "bytes": policy.get("bytes") or (os.path.getsize(stored) if stored else 0),
                "status": "indexed",
                "downloadable": bool(stored),
            }
        )
    for job in workspace.active_jobs():
        documents.append(
            {
                "id": job["id"],
                "name": job["filename"],
                "chunk_count": 0,
                "max_page": 0,
                "uploaded_at": job["created_at"],
                "collection": "General",
                "included": True,
                "citation_required": True,
                "bytes": 0,
                "status": job["status"],
                "downloadable": False,
            }
        )
    return documents


def _require_document(document_id: str) -> None:
    if not any(doc["id"] == document_id for doc in get_all_documents()):
        raise HTTPException(status_code=404, detail="Document not found.")


def _settings_with_drift():
    settings = workspace.settings()
    defaults = env_defaults()
    drift = {
        key: {"saved": settings.get(key), "env_default": defaults.get(key)}
        for key in ("top_k", "similarity_threshold", "rrf_k", "max_parents")
        if settings.get(key) != defaults.get(key)
    }
    return {**settings, "env_defaults": defaults, "settings_drift": drift}


def _eval_summary(suite: str) -> Optional[dict]:
    path = os.path.join(EVAL_REPORTS_DIR, suite, "latest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return {key: report.get(key) for key in ("suite", "tier", "finished_at", "commit", "questions", "metrics", "gate")}


def _analytics_days(days: int) -> int:
    # Not a Literal[...] parameter: query values arrive as strings and would be rejected.
    if days not in (7, 30, 90):
        raise HTTPException(status_code=400, detail="Choose 7, 30, or 90 days.")
    return days


def _sse(payload) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# --- shell, health, and sessions -------------------------------------------------------

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health_check():
    # Public, so it says nothing about models, documents, or configuration.
    return {"status": "ok"}


@app.get("/api/auth/session")
def read_session(request: Request):
    role = request.state.role
    return {
        # In the public demo a visitor is "authenticated" for reading; only the owner can change anything.
        "authenticated": role in {"owner", "visitor"},
        "role": role,
        "demo": security.DEMO_MODE,
        "auth_disabled": security.AUTH_DISABLED,
        "voice": transcribe.enabled(),
        "voice_max_seconds": transcribe.STT_MAX_SECONDS,
    }


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request, response: Response):
    security.rate_limit(request, "login")
    if not security.token_matches(payload.token.strip()):
        log.warning("Failed sign-in from %s", security.client_id(request))
        raise HTTPException(status_code=401, detail="That access token is not valid.")
    security.set_session_cookie(response, request)
    return {"authenticated": True}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(security.SESSION_COOKIE, path="/")
    return {"authenticated": False}


# --- settings and conversations ---------------------------------------------------------

@app.get("/api/settings")
def read_settings():
    return _settings_with_drift()


@app.put("/api/settings")
def write_settings(payload: SettingsPayload):
    workspace.save_settings(payload.model_dump())
    workspace.record_run("Settings updated", "Workspace preferences saved", "Success")
    return _settings_with_drift()


@app.get("/api/conversations")
def read_conversations(request: Request, q: str = Query("", max_length=200)):
    return {"conversations": workspace.list_conversations(q, security.conversation_owner(request))}


@app.post("/api/conversations")
def create_conversation(request: Request):
    return workspace.create_conversation(security.conversation_owner(request))


@app.get("/api/conversations/{conversation_id}")
def read_conversation(request: Request, conversation_id: str = Path(pattern=UUID)):
    if not workspace.conversation_exists(conversation_id, security.conversation_owner(request)):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": workspace.messages(conversation_id)}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(payload: ConversationPayload, request: Request, conversation_id: str = Path(pattern=UUID)):
    renamed = workspace.rename_conversation(conversation_id, payload.title, security.conversation_owner(request))
    if not renamed:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return renamed


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(request: Request, conversation_id: str = Path(pattern=UUID)):
    if not workspace.delete_conversation(conversation_id, security.conversation_owner(request)):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"ok": True}


@app.post("/api/messages/{message_id}/feedback")
def save_feedback(payload: FeedbackPayload, request: Request, message_id: str = Path(pattern=UUID)):
    if not workspace.set_feedback(message_id, payload.rating, security.conversation_owner(request), payload.reason):
        raise HTTPException(status_code=400, detail="Feedback could not be saved for that answer.")
    return {"ok": True, "rating": payload.rating}


@app.post("/api/workspace/reset")
def reset_workspace(payload: ResetPayload):
    if payload.confirm != "DELETE":
        raise HTTPException(status_code=400, detail="Type DELETE to remove the workspace data.")
    for document in get_all_documents():
        delete_document(document["id"])
    workspace.reset()
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return {"ok": True}


# --- index ---------------------------------------------------------------------------------

@app.get("/api/index")
def read_index():
    indexed = [doc for doc in _documents() if doc["status"] == "indexed"]
    return {
        **index_status(),
        "documents": len(indexed),
        "chunks": sum(doc["chunk_count"] or 0 for doc in indexed),
        "storage_bytes": sum(doc["bytes"] or 0 for doc in indexed),
        "runs": workspace.recent_runs(),
        "embedding_dimensions": embedding_dimensions(),
        "stats": workspace.pipeline_stats(),
    }


@app.get("/api/index/versions")
def read_index_versions():
    return {"versions": list_index_versions()}


@app.get("/api/eval/latest")
def read_latest_eval():
    """Summaries of the most recent eval runs (written by `python -m eval`)."""
    return {"hermetic": _eval_summary("hermetic"), "workspace": _eval_summary("workspace")}


@app.post("/api/index/refresh")
def refresh_index(request: Request, publish_override: bool = False):
    security.rate_limit(request, "refresh")
    started = time.perf_counter()
    if publish_override:
        workspace.record_run(
            "Publish override",
            "Publish override requested: the minimum golden-set size check is skipped for this handoff.",
            "Warning",
        )
    try:
        settings = workspace.settings()
        hashes = {
            doc_id: policy.get("content_hash")
            for doc_id, policy in workspace.policies().items()
            if policy.get("content_hash") and not policy.get("deleted")
        }
        status = refresh_active_index(
            "contextual" if settings["contextual_embeddings"] else "raw",
            settings["chunking"],
            hashes,
            publish_override=publish_override,
        )
    except PublishBlocked as exc:
        workspace.record_run("Version handoff", str(exc), "Blocked")
        raise HTTPException(status_code=409, detail={"message": str(exc), "overridable": exc.overridable}) from exc
    except NeedleError as exc:
        workspace.record_run("Version handoff", str(exc), "Failed")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Index refresh failed")
        detail = "Index refresh failed before the new version was published."
        workspace.record_run("Version handoff", detail, "Failed")
        raise HTTPException(status_code=500, detail=detail) from exc
    note = f"Published {status['version_id']} in {int((time.perf_counter() - started) * 1000)} ms"
    workspace.record_run("Version handoff", note + (" (publish_override=true)" if publish_override else ""), "Success")
    return status


@app.post("/api/index/rollback")
def rollback():
    restored = rollback_index()
    if not restored:
        raise HTTPException(status_code=409, detail="There is no earlier index version to restore.")
    workspace.record_run("Rollback", f"Restored {restored['version_id']}", "Success")
    return {"version_id": restored["version_id"], "collection_name": restored["collection_name"]}


@app.post("/api/index/reconcile")
def reconcile():
    removed = 0
    live = {doc["id"] for doc in get_all_documents()}
    policies = workspace.policies()
    for doc_id, policy in policies.items():
        if policy.get("deleted") and doc_id in live:
            delete_document(doc_id)
            removed += 1
    for name in os.listdir(UPLOAD_DIR):
        policy = policies.get(os.path.splitext(name)[0])
        if not policy or policy.get("deleted"):
            os.remove(os.path.join(UPLOAD_DIR, name))
            removed += 1
    orphans = orphaned_documents()
    note = f"Removed {removed} orphaned records"
    if orphans:
        note += f"; {len(orphans)} indexed document(s) have no catalog entry"
    workspace.record_run("Reconcile", note, "Success")
    return {"removed": removed, "orphaned_documents": orphans}


# --- analytics -------------------------------------------------------------------------------

@app.get("/api/analytics")
def read_analytics(request: Request, days: int = Query(30)):
    report = workspace.analytics(_analytics_days(days))
    if request.state.role == "visitor":
        # Aggregate numbers only: other visitors' question text must not be visible.
        report = {**report, "gaps": [], "gaps_no_coverage": [], "gaps_check_failed": []}
    return report


@app.get("/api/analytics/export")
def export_analytics(days: int = Query(30)):
    days = _analytics_days(days)
    report = workspace.analytics(days)
    lines = ["day,questions,grounded"] + [f"{p['day']},{p['questions']},{p['grounded']}" for p in report["series"]]
    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=needle-analytics-{days}d.csv"},
    )


# --- documents -------------------------------------------------------------------------------

@app.post("/api/upload", status_code=202)
def upload_document(request: Request, file: UploadFile = File(...)):
    security.rate_limit(request, "upload")
    filename = security.clean_filename(file.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in security.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a PDF, Word, text, slides, or spreadsheet file.")
    data = security.read_upload(file, request.headers.get("content-length"))
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    security.validate_content(data, ext)
    digest = hashlib.sha256(data).hexdigest()
    existing_id = workspace.find_hash(digest)
    if existing_id:
        return {"id": existing_id, "name": filename, "status": "duplicate",
                "message": "This file is already indexed, so it was not ingested again."}
    if not ingestion.claim(digest):
        raise HTTPException(status_code=409, detail="This file is already being indexed.")
    job_id = workspace.start_job(filename)
    ingestion.submit(job_id=job_id, data=data, filename=filename, ext=ext, content_hash=digest,
                     collection=workspace.settings()["default_collection"])
    return {"job_id": job_id, "name": filename, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str = Path(pattern=UUID)):
    job = workspace.job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload not found.")
    return {key: job[key] for key in ("id", "filename", "status", "error", "document_id", "created_at", "updated_at")}


@app.get("/api/documents")
def list_documents():
    return {"documents": _documents()}


@app.get("/api/documents/{document_id}")
def read_document(
    document_id: str = Path(pattern=UUID),
    offset: int = Query(0, ge=0),
    limit: int = Query(40, ge=1, le=200),
    focus: Optional[str] = Query(None, max_length=200),
):
    match = next((doc for doc in _documents() if doc["id"] == document_id and doc["status"] == "indexed"), None)
    if not match:
        raise HTTPException(status_code=404, detail="Document not found.")
    passages = document_passages(document_id)
    # A citation link names the parent passage it came from; return a page that contains it.
    focus_index = next((i for i, item in enumerate(passages) if focus and item["parent_id"] == focus), None)
    if focus_index is not None:
        offset = 0 if focus_index < 160 else focus_index - 20
        limit = max(limit, focus_index - offset + 10)
    return {
        **match,
        "passages": passages[offset:offset + limit],
        "passage_count": len(passages),
        "offset": offset,
        "focus_index": focus_index,
    }


@app.patch("/api/documents/{document_id}")
def update_document(payload: PolicyPayload, document_id: str = Path(pattern=UUID)):
    _require_document(document_id)
    if not workspace.policy(document_id):
        workspace.set_policy(document_id, collection="General", byte_size=0, stored_name="")
    return workspace.update_policy(document_id, payload.model_dump(exclude_none=True))


@app.get("/api/documents/{document_id}/file")
def download_document(document_id: str = Path(pattern=UUID)):
    if not workspace.settings()["allow_downloads"]:
        raise HTTPException(status_code=403, detail="Source downloads are turned off in settings.")
    policy = workspace.policy(document_id)
    path = _stored_path(policy)
    if not path:
        raise HTTPException(status_code=404, detail="The original file is not stored for this document.")
    name = next((doc["name"] for doc in get_all_documents() if doc["id"] == document_id), os.path.basename(path))
    ext = os.path.splitext(path)[1].lower()
    # Always an attachment with a fixed, non-executable type: a stored file is never rendered as a page.
    return FileResponse(path, filename=security.clean_filename(name), content_disposition_type="attachment",
                        media_type=security.DOWNLOAD_TYPES.get(ext, "application/octet-stream"))


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: str = Path(pattern=UUID)):
    _require_document(document_id)
    policy = workspace.policy(document_id)
    workspace.tombstone(document_id)
    found = delete_document(document_id)
    path = _stored_path(policy)
    if path:
        os.remove(path)
    workspace.forget_document(document_id)
    workspace.record_run("Deletion propagation", f"Removed {document_id}", "Success")
    return {"success": True, "found": found}


# --- voice input --------------------------------------------------------------------------------

@app.post("/api/transcribe")
def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    seconds: float = Form(0),
    language: str = Form(""),
):
    security.rate_limit(request, "voice")
    security.rate_limit(request, "voice_hour")
    if not transcribe.enabled():
        raise HTTPException(status_code=503, detail="Voice input is not configured on this server.")
    declared = int(request.headers.get("content-length") or 0)
    if declared > transcribe.STT_MAX_BYTES * 2:
        raise HTTPException(status_code=413, detail=f"Keep recordings under {transcribe.STT_MAX_SECONDS} seconds.")
    audio = file.file.read(transcribe.STT_MAX_BYTES + 1)
    try:
        text = transcribe.transcribe(audio, seconds, language.strip()[:3])
    except InfraError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except NeedleError as exc:
        status = 429 if "limit" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return {"text": text}


# --- chat ---------------------------------------------------------------------------------------

@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    security.rate_limit(request, "chat")
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if request.state.role == "visitor":
        if len(query) > security.DEMO_MAX_QUESTION_CHARS:
            raise HTTPException(
                status_code=400, detail=f"Demo questions are limited to {security.DEMO_MAX_QUESTION_CHARS} characters."
            )
        security.demo_question_gate(request, workspace.questions_today())
    settings = workspace.settings()
    owner = security.conversation_owner(request)
    conversation_id = workspace.ensure_conversation(payload.conversation_id, query, payload.document_id, owner)
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in workspace.messages(conversation_id)[-HISTORY_TURNS:]
    ]
    user_message_id = workspace.add_message(conversation_id, "user", query, original_query=query)
    options = PipelineOptions(
        top_k=settings["top_k"],
        similarity_threshold=settings["similarity_threshold"],
        rrf_k=settings["rrf_k"],
        max_parents=settings["max_parents"],
        answer_length=settings["answer_length"],
        require_citations=settings["require_citations"],
        citation_style=settings["citation_style"],
        withhold_ungrounded=settings["withhold_ungrounded"],
    )

    def event_stream():
        answer, sources, validation, trace, result, failed = [], [], None, None, {}, None
        yield _sse({"type": "conversation", "id": conversation_id})
        try:
            for event in run_pipeline(
                query,
                history=history,
                document_id=payload.document_id,
                exclude_document_ids=workspace.excluded_document_ids(),
                citation_required_ids=workspace.citation_required_ids(),
                options=options,
            ):
                kind = event.get("type")
                if kind == "result":
                    result = event
                    continue
                if kind == "chunk":
                    answer.append(event.get("content") or "")
                elif kind == "sources":
                    sources = event.get("data") or []
                elif kind == "validation":
                    validation = event
                elif kind == "trace":
                    trace = event
                elif kind == "error":
                    failed = event.get("content")
                yield _sse(event)
        except Exception:
            log.exception("Chat stream failed")
            failed = "The answer pipeline failed."
            yield _sse({"type": "error", "content": failed})
        standalone = ((result.get("retrieval") or {}).get("standalone_query") or "").strip()
        if standalone and standalone != query:
            workspace.set_rewritten_query(user_message_id, standalone)
        latencies = result.get("latencies_ms") or {}
        if trace is not None:
            # The trace event goes out before the answer is written; store the final timings.
            trace = {**trace, "latencies_ms": latencies or trace.get("latencies_ms")}
        content = "".join(answer) or (failed or "")
        message_id = workspace.add_message(conversation_id, "assistant", content, sources, validation, trace)
        scores = [source.get("vector_similarity") for source in sources if source.get("vector_similarity") is not None]
        category = result.get("reject_category")
        declined = bool(result.get("declined"))
        if result.get("passed") and not declined:
            outcome = "answered"
        elif failed or category in {"infra_error", "pipeline_error"}:
            outcome = "error"
        elif declined or category in {"retrieval_abstain", "empty_question", "injection_scan"}:
            outcome = "no_coverage"
        else:
            outcome = "check_failed"
        released = bool(result.get("released"))
        workspace.record_event(
            query=query,
            grounded=bool(result.get("passed")) and not declined,
            withheld=not released or declined,
            outcome=outcome,
            latency_ms=int(latencies.get("total") or 0),
            retrieval_ms=int((latencies.get("retrieve") or 0) + (latencies.get("rerank") or 0)),
            source_count=len(sources) if released else 0,
            best_similarity=max(scores) if scores else None,
            relevance=(result.get("retrieval") or {}).get("relevance"),
            cited=released and bool(re.search(r"\[\d+\]", content)),
            candidate_count=(trace or {}).get("candidates") or 0,
            reject_category=category,
        )
        yield _sse({"type": "saved", "message_id": message_id, "conversation_id": conversation_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    # Loopback by default. Set NEEDLE_HOST=0.0.0.0 (and NEEDLE_ALLOWED_HOSTS) to serve a network.
    host = env_str("NEEDLE_HOST", "127.0.0.1")
    port = env_int("NEEDLE_PORT", 8000)
    print(f"\nNeedle is running at http://{'localhost' if host in {'127.0.0.1', '0.0.0.0'} else host}:{port}\n")
    uvicorn.run(app, host=host, port=port)
