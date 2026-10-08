"""The active index: Chroma collections, the parent/keyword store, and read access to documents and versions."""

import hashlib
import logging
import os
import threading
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import chromadb

from config import (
    CHECKER_MODEL,
    EVAL_MIN_GOLDEN,
    JEV_CANDIDATE_LIMIT,
    JEV_MODEL,
    JEV_RELEVANCE_THRESHOLD,
    MAX_PARENTS,
    OPENROUTER_MODEL,
    RETRIEVAL_RERANK_MODE,
    RETRY_THRESHOLD,
    RRF_K,
    TOP_K_CHILDREN,
    VECTOR_SIMILARITY_THRESHOLD,
)
from embedder import configured_embedding_model
from index_store import IndexStore
from llm import writer_breaker
from paths import data_path
from pipeline_logic import IncompatibleIndex
from rerank import jev_breaker, jev_configured, last_rerank_mode

log = logging.getLogger("needle")
STORE_DIR = data_path("chroma_store")
os.makedirs(STORE_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=STORE_DIR)
# Parents, keyword index, document catalog, and the index version registry.
store = IndexStore(os.path.join(STORE_DIR, "bm25_index.db"))
# Serialises writes to the index (uploads, deletes, refreshes); reads do not take it.
mutation_lock = threading.Lock()


def require_compatible() -> Dict[str, Any]:
    expected = configured_embedding_model()
    active = store.ensure_active_version(expected)
    if active["embedding_model"] != expected:
        raise IncompatibleIndex(
            "The active index was built with "
            f"{active['embedding_model']}, which does not match {expected}. "
            "Refresh the index before searching it."
        )
    return active


def env_defaults() -> Dict[str, Any]:
    """Process env defaults. Workspace settings may override these at query time."""
    return {
        "top_k": TOP_K_CHILDREN,
        "similarity_threshold": VECTOR_SIMILARITY_THRESHOLD,
        "rrf_k": RRF_K,
        "max_parents": MAX_PARENTS,
        "jev_relevance_threshold": JEV_RELEVANCE_THRESHOLD,
        "jev_candidate_limit": JEV_CANDIDATE_LIMIT or None,
        "retry_threshold": RETRY_THRESHOLD,
        "retrieval_rerank_mode": RETRIEVAL_RERANK_MODE,
        "answer_model": OPENROUTER_MODEL,
        "checker_model": CHECKER_MODEL,
        "jev_model": JEV_MODEL,
        "chunking": os.getenv("CHUNKING_STRATEGY", "Parent-child"),
        "eval_min_golden": EVAL_MIN_GOLDEN,
    }


def index_status() -> Dict[str, Any]:
    expected = configured_embedding_model()
    active = store.ensure_active_version(expected)
    return {
        "version_id": active["version_id"],
        "collection_name": active["collection_name"],
        "embedding_model": active["embedding_model"],
        "compatible": active["embedding_model"] == expected,
        "expected_embedding_model": expected,
        "jev_configured": jev_configured(),
        "jev_model": JEV_MODEL,
        "answer_configured": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
        "answer_model": OPENROUTER_MODEL,
        "checker_model": CHECKER_MODEL,
        "top_k": TOP_K_CHILDREN,
        "similarity_threshold": VECTOR_SIMILARITY_THRESHOLD,
        "jev_relevance_threshold": JEV_RELEVANCE_THRESHOLD,
        "retry_threshold": RETRY_THRESHOLD,
        "rrf_k": RRF_K,
        "max_parents": MAX_PARENTS,
        "embed_style": active.get("embed_style") or "raw",
        "chunking": active.get("chunking") or os.getenv("CHUNKING_STRATEGY", "Parent-child"),
        "index_status": active["status"],
        "jev_circuit": jev_breaker.status(),
        "writer_circuit": writer_breaker.status(),
        "rerank_mode": last_rerank_mode(),
        "env_defaults": env_defaults(),
        "eval_min_golden": EVAL_MIN_GOLDEN,
    }


def collection_for(name: str):
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


def collection_model(collection_name: str) -> Optional[str]:
    """The embedding model recorded for a collection in the version registry."""
    for version in store.list_versions():
        if version["collection_name"] == collection_name:
            return version["embedding_model"]
    return None


def active_collection():
    active = require_compatible()
    return collection_for(active["collection_name"]), active


def parent_id(chunk_id: str, metadata: Dict[str, Any]) -> str:
    explicit = metadata.get("parent_id")
    if explicit:
        return str(explicit)
    parts = (chunk_id or "").split("_")
    if len(parts) >= 2 and parts[0] and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return f"{metadata.get('document_id', '')}_{metadata.get('chunk_index', 0)}"


def get_all_documents() -> List[Dict[str, Any]]:
    return store.list_documents()


def delete_document(document_id: str) -> bool:
    with mutation_lock:
        return _delete_document(document_id)


def _delete_document(document_id: str) -> bool:
    existed = store.document_exists(document_id)
    for version in store.list_versions():
        if version["status"] == "failed":
            continue
        try:
            collection = chroma_client.get_collection(version["collection_name"])
            collection.delete(where={"document_id": document_id})
        except Exception:
            log.warning("Could not delete %s from index %s", document_id, version["collection_name"], exc_info=True)
    if existed:
        store.mark_deleted(document_id)
    return existed


def document_hash_map(collection_name: str, content_hashes: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Map document_id -> content_hash using metadata, caller hashes, or parent-text fingerprint."""
    collection = chroma_client.get_collection(collection_name)
    payload = collection.get(include=["metadatas", "documents"])
    by_doc_texts: Dict[str, List[str]] = {}
    by_doc_hash: Dict[str, str] = {}
    for metadata, document in zip(payload.get("metadatas") or [], payload.get("documents") or []):
        metadata = metadata or {}
        document_id = metadata.get("document_id") or ""
        if not document_id:
            continue
        explicit = (content_hashes or {}).get(document_id) or metadata.get("content_hash") or ""
        if explicit:
            by_doc_hash[document_id] = str(explicit)
        text = metadata.get("parent_text") or metadata.get("raw_text") or document or ""
        by_doc_texts.setdefault(document_id, []).append(text)
    for document_id, texts in by_doc_texts.items():
        if document_id not in by_doc_hash:
            blob = "\n".join(sorted((text or "").strip() for text in texts if (text or "").strip()))
            by_doc_hash[document_id] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return by_doc_hash


def list_parent_passages(collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Unique parent passages from the active (or named) index, with migration-safe content hashes."""
    if collection_name:
        collection = chroma_client.get_collection(collection_name)
    else:
        collection, _active = active_collection()
        collection_name = collection.name
    hash_map = document_hash_map(collection_name)
    payload = collection.get(include=["metadatas", "documents"])
    rows = list(zip(payload.get("ids") or [], payload.get("documents") or [], payload.get("metadatas") or []))
    parents = store.fetch_parents(parent_id(chunk_id, metadata or {}) for chunk_id, _text, metadata in rows)
    grouped: Dict[str, Dict[str, Any]] = {}
    for chunk_id, text, metadata in rows:
        metadata = metadata or {}
        parent_key = parent_id(chunk_id, metadata)
        document_id = metadata.get("document_id") or ""
        parent_text = metadata.get("parent_text") or metadata.get("raw_text") or text or ""
        stored = parents.get(parent_key)
        if stored and stored.get("parent_text"):
            parent_text = stored["parent_text"]
        slot = grouped.setdefault(
            parent_key,
            {
                "parent_id": parent_key,
                "document_id": document_id,
                "document_name": metadata.get("document_name") or "",
                "page_number": int(metadata.get("page_number") or 1),
                "header_context": metadata.get("header_context") or "",
                "text": parent_text,
                "content_hash": hash_map.get(document_id, ""),
            },
        )
        if len(parent_text) > len(slot["text"] or ""):
            slot["text"] = parent_text
    return [item for item in grouped.values() if (item.get("text") or "").strip()]


def orphaned_documents() -> List[Dict[str, Any]]:
    """Documents with vectors in the active index but no catalog or keyword-index entry.

    They are searchable but cannot be listed or deleted from the UI. Reported, never auto-deleted.
    """
    collection, _active = active_collection()
    listed = {doc["id"] for doc in store.list_documents()}
    counts: Dict[str, Dict[str, Any]] = {}
    for meta in collection.get(include=["metadatas"]).get("metadatas") or []:
        document_id = (meta or {}).get("document_id") or ""
        if document_id and document_id not in listed:
            slot = counts.setdefault(document_id, {"document_id": document_id, "document_name": meta.get("document_name") or "", "chunks": 0})
            slot["chunks"] += 1
    return sorted(counts.values(), key=lambda item: item["document_name"])


def list_index_versions() -> List[Dict[str, Any]]:
    """Registry rows, newest first, with the vector count of each collection that still exists."""
    versions = []
    for row in reversed(store.list_versions()):
        try:
            count = chroma_client.get_collection(row["collection_name"]).count()
        except Exception:
            count = None
        versions.append({**row, "chunk_count": count})
    return versions


def load_parent(child: Dict[str, Any]) -> Dict[str, Any]:
    stored = store.fetch_parent(child["parent_id"])
    if stored:
        return {
            "parent_id": child.get("parent_id") or stored.get("parent_id"),
            "text": stored["parent_text"],
            "page_number": int(stored["page_number"] or 1),
            "chunk_index": int(stored["chunk_index"] or 0),
            "document_id": stored["document_id"],
            "document_name": stored["document_name"],
            "header_context": stored["header_context"] or "",
            "similarity_score": round(float(child["jev_score"]), 4),
            "vector_similarity": child.get("similarity"),
            "matched_passage": child["text"],
            "unit_type": child.get("unit_type") or "child",
            "relation": child.get("relation") or "supporting",
        }
    parent_text = child.get("parent_text") or child["text"]
    return {
        "parent_id": child.get("parent_id"),
        "text": parent_text,
        "page_number": int(child.get("page_number") or 1),
        "chunk_index": int(child.get("chunk_index") or 0),
        "document_id": child.get("document_id", ""),
        "document_name": child.get("document_name", ""),
        "header_context": child.get("header_context") or "",
        "similarity_score": round(float(child["jev_score"]), 4),
        "vector_similarity": child.get("similarity"),
        "matched_passage": child["text"],
        "unit_type": child.get("unit_type") or "child",
        "relation": child.get("relation") or "supporting",
    }


def document_passages(document_id: str) -> List[Dict[str, Any]]:
    """Parent passages stored for one document, used by the document detail view."""
    collection, _active = active_collection()
    try:
        data = collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
    except Exception:
        log.exception("Could not read passages for %s", document_id)
        return []
    grouped: Dict[str, Dict[str, Any]] = {}
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    parents = store.fetch_parents(parent_id(chunk_id, metadata or {}) for chunk_id, metadata in zip(ids, metas))
    for chunk_id, text, metadata in zip(ids, docs, metas):
        metadata = metadata or {}
        parent_key = parent_id(chunk_id, metadata)
        slot = grouped.setdefault(
            parent_key,
            {
                "parent_id": parent_key,
                "page_number": int(metadata.get("page_number") or 1),
                "header_context": metadata.get("header_context") or "",
                "text": "",
            },
        )
        stored = parents.get(parent_key)
        if stored and stored.get("parent_text"):
            slot["text"] = stored["parent_text"]
            slot["page_number"] = int(stored.get("page_number") or slot["page_number"])
            slot["header_context"] = stored.get("header_context") or slot["header_context"]
        elif metadata.get("parent_text"):
            slot["text"] = metadata["parent_text"]
        elif text and not slot["text"]:
            slot["text"] = text
    passages = [item for item in grouped.values() if item["text"]]
    passages.sort(key=lambda item: (item["page_number"], item["parent_id"]))
    return passages
