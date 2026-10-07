import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_engine import (
    CHUNKING_STRATEGIES,
    NeedleError,
    PipelineOptions,
    PublishBlocked,
    delete_document,
    document_passages,
    embedding_dimensions,
    env_defaults,
    get_all_documents,
    index_status,
    list_index_versions,
    orphaned_documents,
    process_document,
    refresh_active_index,
    rollback_index,
    run_pipeline,
)
from paths import data_path
from workspace import WorkspaceStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("needle")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".text", ".docx", ".pptx", ".xlsx", ".csv"}
ROOT = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(ROOT, "..", "frontend")
UPLOAD_DIR = data_path("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

workspace = WorkspaceStore(data_path("workspace.db"))
EVAL_REPORTS_DIR = os.path.join(ROOT, "eval", "reports")
_hits: dict = defaultdict(list)

app = FastAPI(title="Needle", description="Source-grounded knowledge workspace", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Handlers below are plain `def` on purpose. They call SQLite, Chroma, the
# embedder, and OpenRouter synchronously; FastAPI runs `def` handlers in a
# thread pool, so a long upload or refresh no longer freezes every other request.


class ChatRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    conversation_id: Optional[str] = None


class SettingsPayload(BaseModel):
    workspace_name: str = "Needle"
    profile_name: str = "Operator"
    default_collection: str = "General"
    region: str = "European Union"
    timezone: str = "UTC"
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
    collection: Optional[str] = None


class FeedbackPayload(BaseModel):
    rating: str


class InvitePayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=3, max_length=120)
    role: str = "Member"


class ResetPayload(BaseModel):
    confirm: str


def _allow(key: str, limit: int = 40) -> None:
    now = time.time()
    recent = [stamp for stamp in _hits[key] if now - stamp < 60]
    if len(recent) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests. Wait a moment and try again.")
    recent.append(now)
    _hits[key] = recent


def _stored_path(document_id: str) -> Optional[str]:
    policy = workspace.policy(document_id)
    if not policy or not policy.get("stored_name"):
        return None
    path = os.path.join(UPLOAD_DIR, policy["stored_name"])
    return path if os.path.exists(path) else None


def _documents():
    policies = workspace.policies()
    jobs = workspace.active_jobs()
    documents = []
    for doc in get_all_documents():
        policy = policies.get(doc["id"], {})
        stored = _stored_path(doc["id"])
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
    for job in jobs:
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


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Needle", **index_status()}


def _settings_with_drift():
    settings = workspace.settings()
    defaults = env_defaults()
    drift = {}
    for key in ("top_k", "similarity_threshold", "rrf_k", "max_parents"):
        saved = settings.get(key)
        env_value = defaults.get(key)
        if saved != env_value:
            drift[key] = {"saved": saved, "env_default": env_value}
    return {
        **settings,
        "env_defaults": defaults,
        "settings_drift": drift,
    }


@app.get("/api/settings")
def read_settings():
    return _settings_with_drift()


@app.put("/api/settings")
def write_settings(payload: SettingsPayload):
    workspace.save_settings(payload.model_dump())
    workspace.record_run("Settings updated", "Workspace preferences saved", "Success")
    return _settings_with_drift()


@app.get("/api/conversations")
def read_conversations(q: str = ""):
    return {"conversations": workspace.list_conversations(q)}


@app.post("/api/conversations")
def create_conversation():
    return workspace.create_conversation()


@app.get("/api/conversations/{conversation_id}")
def read_conversation(conversation_id: str):
    messages = workspace.messages(conversation_id)
    if not messages and not any(item["id"] == conversation_id for item in workspace.list_conversations()):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": messages}


@app.post("/api/messages/{message_id}/feedback")
def save_feedback(message_id: str, payload: FeedbackPayload):
    if not workspace.set_feedback(message_id, payload.rating):
        raise HTTPException(status_code=400, detail="Feedback could not be saved for that answer.")
    return {"ok": True, "rating": payload.rating}


@app.get("/api/members")
def read_members():
    return {"members": workspace.list_members()}


@app.post("/api/members")
def invite_member(payload: InvitePayload):
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", payload.email.strip()):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    try:
        member = workspace.invite_member(payload.name, payload.email, payload.role)
    except Exception as exc:
        log.info("Invite failed: %s", exc)
        raise HTTPException(status_code=409, detail="A member with that email is already in the workspace.") from exc
    workspace.record_run("Member invited", f"{member['name']} joined as {member['role']}", "Success")
    return member


@app.get("/api/integrations")
def read_integrations():
    return {
        "integrations": [
            {"id": "files", "name": "File upload", "connected": True, "detail": "PDF, Word, text, slides, and spreadsheets"},
            {"id": "drive", "name": "Google Drive", "connected": False, "detail": "Not configured on this server"},
            {"id": "notion", "name": "Notion", "connected": False, "detail": "Not configured on this server"},
            {"id": "slack", "name": "Slack", "connected": False, "detail": "Not configured on this server"},
        ]
    }


@app.post("/api/workspace/reset")
def reset_workspace(payload: ResetPayload):
    if payload.confirm != "DELETE":
        raise HTTPException(status_code=400, detail="Type DELETE to remove the workspace data.")
    for document in get_all_documents():
        delete_document(document["id"])
        path = _stored_path(document["id"])
        if path and os.path.exists(path):
            os.remove(path)
    workspace.reset()
    if os.path.isdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    return {"ok": True}


@app.get("/api/index")
def read_index():
    status = index_status()
    documents = _documents()
    indexed = [doc for doc in documents if doc["status"] == "indexed"]
    return {
        **status,
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


@app.get("/api/eval/latest")
def read_latest_eval():
    """Summaries of the most recent eval runs (written by `python -m eval`)."""
    return {"hermetic": _eval_summary("hermetic"), "workspace": _eval_summary("workspace")}


@app.post("/api/index/refresh")
def refresh_index(publish_override: bool = False):
    started = time.perf_counter()
    if publish_override:
        workspace.record_run(
            "Publish override",
            "Publish override requested: the minimum golden-set size check is skipped for this handoff.",
            "Warning",
        )
        log.warning("Index refresh running with publish_override=true")
    try:
        settings = workspace.settings()
        style = "contextual" if settings["contextual_embeddings"] else "raw"
        hashes = {
            doc_id: policy.get("content_hash")
            for doc_id, policy in workspace.policies().items()
            if policy.get("content_hash") and not policy.get("deleted")
        }
        status = refresh_active_index(
            style,
            settings["chunking"],
            hashes,
            publish_override=publish_override,
        )
    except PublishBlocked as exc:
        workspace.record_run("Version handoff", str(exc), "Blocked")
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "overridable": exc.overridable},
        ) from exc
    except NeedleError as exc:
        workspace.record_run("Version handoff", str(exc), "Failed")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Index refresh failed")
        detail = "Index refresh failed before the new version was published."
        workspace.record_run("Version handoff", detail, "Failed")
        raise HTTPException(status_code=500, detail=detail) from exc
    elapsed = int((time.perf_counter() - started) * 1000)
    note = f"Published {status['version_id']} in {elapsed} ms"
    if publish_override:
        note += " (publish_override=true)"
    workspace.record_run("Version handoff", note, "Success")
    return status


@app.get("/api/analytics")
def read_analytics(days: int = 30):
    if days not in {7, 30, 90}:
        raise HTTPException(status_code=400, detail="Choose 7, 30, or 90 days.")
    return workspace.analytics(days)


@app.get("/api/analytics/export")
def export_analytics(days: int = 30):
    report = workspace.analytics(days if days in {7, 30, 90} else 30)
    lines = ["day,questions,grounded"]
    for point in report["series"]:
        lines.append(f"{point['day']},{point['questions']},{point['grounded']}")
    return StreamingResponse(
        iter(["\n".join(lines) + "\n"]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=needle-analytics-{report['days']}d.csv"},
    )


@app.post("/api/upload")
def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "document")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a PDF, Word, text, slides, or spreadsheet file.")
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File is larger than 50 MB.")
    digest = hashlib.sha256(file_bytes).hexdigest()
    existing_id = workspace.find_hash(digest)
    if existing_id:
        return {
            "id": existing_id,
            "name": filename,
            "duplicate": True,
            "message": "This file is already indexed, so it was not ingested again.",
        }
    settings = workspace.settings()
    job_id = workspace.start_job(filename)
    workspace.update_job(job_id, "processing")
    try:
        doc = process_document(file_bytes, filename, file.content_type or "", content_hash=digest)
    except Exception as exc:
        log.exception("Upload failed")
        message = str(exc) if isinstance(exc, (ValueError, NeedleError)) else "The file could not be indexed."
        workspace.update_job(job_id, "failed", message)
        workspace.record_run("Ingestion", f"{filename} failed", "Failed")
        raise HTTPException(status_code=400, detail=message) from exc
    stored_name = f"{doc.id}{ext}"
    with open(os.path.join(UPLOAD_DIR, stored_name), "wb") as handle:
        handle.write(file_bytes)
    workspace.set_policy(
        doc.id,
        collection=settings["default_collection"],
        byte_size=len(file_bytes),
        stored_name=stored_name,
        content_hash=digest,
    )
    workspace.update_job(job_id, "completed", document_id=doc.id)
    workspace.record_run("Ingestion", f"{filename}: {doc.num_chunks} chunks", "Success")
    return {
        "id": doc.id,
        "name": doc.name,
        "num_chunks": doc.num_chunks,
        "num_pages": doc.num_pages,
        "file_type": doc.file_type,
        "uploaded_at": doc.uploaded_at,
    }


@app.get("/api/documents")
def list_documents():
    return {"documents": _documents()}


@app.get("/api/documents/{document_id}")
def read_document(document_id: str):
    match = next((doc for doc in _documents() if doc["id"] == document_id and doc["status"] == "indexed"), None)
    if not match:
        raise HTTPException(status_code=404, detail="Document not found.")
    passages = document_passages(document_id)
    return {**match, "passages": passages[:40], "passage_count": len(passages)}


@app.patch("/api/documents/{document_id}")
def update_document(document_id: str, payload: PolicyPayload):
    if not any(doc["id"] == document_id for doc in get_all_documents()):
        raise HTTPException(status_code=404, detail="Document not found.")
    if not workspace.policy(document_id):
        workspace.set_policy(document_id, collection="General", byte_size=0, stored_name="")
    updated = workspace.update_policy(document_id, payload.model_dump(exclude_none=True))
    return updated


@app.get("/api/documents/{document_id}/file")
def download_document(document_id: str):
    if not workspace.settings()["allow_downloads"]:
        raise HTTPException(status_code=403, detail="Source downloads are turned off in settings.")
    path = _stored_path(document_id)
    if not path:
        raise HTTPException(status_code=404, detail="The original file is not stored for this document.")
    policy = workspace.policy(document_id) or {}
    return FileResponse(path, filename=policy.get("stored_name") or os.path.basename(path))


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: str):
    if not any(doc["id"] == document_id for doc in get_all_documents()):
        raise HTTPException(status_code=404, detail="Document not found.")
    workspace.tombstone(document_id)
    found = delete_document(document_id)
    path = _stored_path(document_id)
    if path and os.path.exists(path):
        os.remove(path)
    workspace.forget_document(document_id)
    if not found:
        raise HTTPException(status_code=404, detail="Document not found.")
    workspace.record_run("Deletion propagation", f"Removed {document_id}", "Success")
    return {"success": True, "found": found}


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
    for doc_id, policy in workspace.policies().items():
        if policy.get("deleted") and doc_id in live:
            delete_document(doc_id)
            removed += 1
    for name in os.listdir(UPLOAD_DIR):
        document_id = os.path.splitext(name)[0]
        policy = workspace.policy(document_id)
        if not policy or policy.get("deleted"):
            os.remove(os.path.join(UPLOAD_DIR, name))
            removed += 1
    orphans = orphaned_documents()
    note = f"Removed {removed} orphaned records"
    if orphans:
        note += f"; {len(orphans)} indexed document(s) have no catalog entry"
    workspace.record_run("Reconcile", note, "Success")
    return {"removed": removed, "orphaned_documents": orphans}


@app.post("/api/chat")
def chat(request: ChatRequest):
    _allow("chat")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 4000:
        raise HTTPException(status_code=400, detail="Question is too long.")
    settings = workspace.settings()
    conversation_id = workspace.ensure_conversation(request.conversation_id, query, request.document_id)
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in workspace.messages(conversation_id)
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
        answer = []
        sources = []
        validation = None
        trace = None
        result = {}
        failed = None
        yield _sse({"type": "conversation", "id": conversation_id})
        try:
            for event in run_pipeline(
                query,
                history=history,
                document_id=request.document_id,
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
        if result.get("passed") and not result.get("declined"):
            outcome = "answered"
        elif result.get("declined"):
            outcome = "no_coverage"
        elif failed or category in {"infra_error", "pipeline_error"}:
            outcome = "error"
        elif category in {"retrieval_abstain", "empty_question", "injection_scan"}:
            outcome = "no_coverage"
        else:
            outcome = "check_failed"
        released = bool(result.get("released"))
        workspace.record_event(
            query=query,
            grounded=bool(result.get("passed")) and not result.get("declined"),
            withheld=not released or bool(result.get("declined")),
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
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _sse(payload) -> str:
    return f"data: {json.dumps(payload)}\n\n"


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("Server starting. Open this link in your browser:")
    print("http://localhost:8000")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
