import json
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_engine import (
    NeedleError,
    delete_document,
    document_passages,
    generate_answer_stream,
    get_all_documents,
    index_status,
    condense_query,
    process_document,
    refresh_active_index,
)
from pipeline_logic import needs_condense
from workspace import WorkspaceStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("needle")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".text", ".docx", ".pptx", ".xlsx", ".csv"}
ROOT = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(ROOT, "..", "frontend")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

workspace = WorkspaceStore(os.path.join(ROOT, "workspace.db"))
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
    answer_length: str = "Balanced"
    citation_style: str = "Inline numbered"
    require_citations: bool = True
    withhold_ungrounded: bool = True
    top_k: int = 30
    similarity_threshold: float = 0.30
    max_parents: int = 5
    rrf_k: int = 60
    contextual_embeddings: bool = True
    chunking: str = "Parent-child"


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
async def health_check():
    return {"status": "healthy", "service": "Needle", **index_status()}


@app.get("/api/settings")
async def read_settings():
    return workspace.settings()


@app.put("/api/settings")
async def write_settings(payload: SettingsPayload):
    saved = workspace.save_settings(payload.model_dump())
    workspace.record_run("Settings updated", "Workspace preferences saved", "Success")
    return saved


@app.get("/api/conversations")
async def read_conversations(q: str = ""):
    return {"conversations": workspace.list_conversations(q)}


@app.post("/api/conversations")
async def create_conversation():
    return workspace.create_conversation()


@app.get("/api/conversations/{conversation_id}")
async def read_conversation(conversation_id: str):
    messages = workspace.messages(conversation_id)
    if not messages and not any(item["id"] == conversation_id for item in workspace.list_conversations()):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": messages}


@app.post("/api/messages/{message_id}/feedback")
async def save_feedback(message_id: str, payload: FeedbackPayload):
    if not workspace.set_feedback(message_id, payload.rating):
        raise HTTPException(status_code=400, detail="Feedback could not be saved for that answer.")
    return {"ok": True, "rating": payload.rating}


@app.get("/api/members")
async def read_members():
    return {"members": workspace.list_members()}


@app.post("/api/members")
async def invite_member(payload: InvitePayload):
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
async def read_integrations():
    return {
        "integrations": [
            {"id": "files", "name": "File upload", "connected": True, "detail": "PDF, Word, text, slides, and spreadsheets"},
            {"id": "drive", "name": "Google Drive", "connected": False, "detail": "Not configured on this server"},
            {"id": "notion", "name": "Notion", "connected": False, "detail": "Not configured on this server"},
            {"id": "slack", "name": "Slack", "connected": False, "detail": "Not configured on this server"},
        ]
    }


@app.post("/api/workspace/reset")
async def reset_workspace(payload: ResetPayload):
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
async def read_index():
    status = index_status()
    documents = _documents()
    indexed = [doc for doc in documents if doc["status"] == "indexed"]
    return {
        **status,
        "documents": len(indexed),
        "chunks": sum(doc["chunk_count"] or 0 for doc in indexed),
        "storage_bytes": sum(doc["bytes"] or 0 for doc in indexed),
        "runs": workspace.recent_runs(),
        "embedding_dimensions": 384,
    }


@app.post("/api/index/refresh")
async def refresh_index():
    started = time.perf_counter()
    try:
        style = "contextual" if workspace.settings()["contextual_embeddings"] else "raw"
        status = refresh_active_index(style)
    except NeedleError as exc:
        workspace.record_run("Version handoff", str(exc), "Failed")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Index refresh failed")
        workspace.record_run("Version handoff", "Refresh failed before publish", "Failed")
        raise HTTPException(status_code=500, detail="Index refresh failed before the new version was published.") from exc
    elapsed = int((time.perf_counter() - started) * 1000)
    workspace.record_run("Version handoff", f"Published {status['version_id']} in {elapsed} ms", "Success")
    return status


@app.get("/api/analytics")
async def read_analytics(days: int = 30):
    if days not in {7, 30, 90}:
        raise HTTPException(status_code=400, detail="Choose 7, 30, or 90 days.")
    return workspace.analytics(days)


@app.get("/api/analytics/export")
async def export_analytics(days: int = 30):
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
async def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "document")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a PDF, Word, text, slides, or spreadsheet file.")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File is larger than 50 MB.")
    settings = workspace.settings()
    job_id = workspace.start_job(filename)
    workspace.update_job(job_id, "processing")
    try:
        doc = process_document(file_bytes, filename, file.content_type or "", settings["chunking"])
    except Exception as exc:
        log.exception("Upload failed")
        message = str(exc) if isinstance(exc, ValueError) else "The file could not be indexed."
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
async def list_documents():
    return {"documents": _documents()}


@app.get("/api/documents/{document_id}")
async def read_document(document_id: str):
    match = next((doc for doc in _documents() if doc["id"] == document_id and doc["status"] == "indexed"), None)
    if not match:
        raise HTTPException(status_code=404, detail="Document not found.")
    passages = document_passages(document_id)
    preview = "\n\n".join(item["text"] for item in passages[:6])
    return {**match, "passages": passages[:12], "preview": preview[:8000]}


@app.patch("/api/documents/{document_id}")
async def update_document(document_id: str, payload: PolicyPayload):
    if not any(doc["id"] == document_id for doc in get_all_documents()):
        raise HTTPException(status_code=404, detail="Document not found.")
    if not workspace.policy(document_id):
        workspace.set_policy(document_id, collection="General", byte_size=0, stored_name="")
    updated = workspace.update_policy(document_id, payload.model_dump(exclude_none=True))
    return updated


@app.get("/api/documents/{document_id}/file")
async def download_document(document_id: str):
    if not workspace.settings()["allow_downloads"]:
        raise HTTPException(status_code=403, detail="Source downloads are turned off in settings.")
    path = _stored_path(document_id)
    if not path:
        raise HTTPException(status_code=404, detail="The original file is not stored for this document.")
    policy = workspace.policy(document_id) or {}
    return FileResponse(path, filename=policy.get("stored_name") or os.path.basename(path))


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str):
    found = delete_document(document_id)
    path = _stored_path(document_id)
    if path and os.path.exists(path):
        os.remove(path)
    workspace.forget_document(document_id)
    if not found:
        raise HTTPException(status_code=404, detail="Document not found.")
    workspace.record_run("Deletion propagation", f"Removed {document_id}", "Success")
    return {"success": True}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    _allow("chat")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 4000:
        raise HTTPException(status_code=400, detail="Question is too long.")
    settings = workspace.settings()
    conversation_id = workspace.ensure_conversation(request.conversation_id, query, request.document_id)
    history = workspace.messages(conversation_id)
    rewritten = None
    search_query = query
    if needs_condense(history):
        try:
            rewritten = condense_query(history, query)
            search_query = rewritten or query
        except NeedleError as exc:
            log.warning("Query rewrite skipped: %s", exc)
    workspace.add_message(conversation_id, "user", query, original_query=query, rewritten_query=rewritten)
    started = time.perf_counter()

    def event_stream():
        answer = []
        sources = []
        validation = None
        trace = None
        withheld = False
        failed = None
        yield f"data: {json.dumps({'type': 'conversation', 'id': conversation_id})}\n\n"
        try:
            for data in generate_answer_stream(
                query=search_query,
                document_id=request.document_id,
                top_k=settings["top_k"],
                similarity_threshold=settings["similarity_threshold"],
                rrf_k=settings["rrf_k"],
                max_parents=settings["max_parents"],
                exclude_document_ids=workspace.excluded_document_ids(),
                answer_length=settings["answer_length"],
                require_citations=settings["require_citations"],
                citation_style=settings["citation_style"],
                withhold_ungrounded=settings["withhold_ungrounded"],
            ):
                payload = json.loads(data)
                if payload.get("type") == "chunk":
                    answer.append(payload.get("content") or "")
                elif payload.get("type") == "sources":
                    sources = payload.get("data") or []
                elif payload.get("type") == "validation":
                    validation = payload
                    withheld = not payload.get("passed") and settings["withhold_ungrounded"]
                elif payload.get("type") == "trace":
                    trace = payload
                elif payload.get("type") == "error":
                    failed = payload.get("content")
                yield f"data: {data}\n\n"
        except Exception:
            log.exception("Chat stream failed")
            failed = "The answer pipeline failed."
            yield f"data: {json.dumps({'type': 'error', 'content': failed})}\n\n"
        content = "".join(answer) or (failed or "")
        message_id = workspace.add_message(conversation_id, "assistant", content, sources, validation, trace)
        best = None
        if sources:
            scores = [source.get("vector_similarity") for source in sources if source.get("vector_similarity") is not None]
            best = max(scores) if scores else None
        workspace.record_event(
            query=query,
            grounded=bool(validation and validation.get("passed")),
            withheld=withheld or bool(failed),
            latency_ms=int((time.perf_counter() - started) * 1000),
            source_count=len(sources),
            best_similarity=best,
            candidate_count=(trace or {}).get("candidates") or 0,
        )
        yield f"data: {json.dumps({'type': 'saved', 'message_id': message_id, 'conversation_id': conversation_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("Server starting. Open this link in your browser:")
    print("http://localhost:8000")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
