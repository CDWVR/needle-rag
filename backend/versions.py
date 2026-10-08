"""Index versions: build a new collection, gate it on golden-set recall, publish, and roll back."""

import logging
import os
import time
import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from config import (
    EVAL_MIN_GOLDEN,
    REFRESH_GATE_USE_JEV,
    WORKSPACE_GOLDEN,
    env_float,
    env_int,
)
from catalog import (
    chroma_client,
    collection_for,
    document_hash_map,
    index_status,
    mutation_lock,
    store,
)
from embedder import configured_embedding_model, embed
from pipeline import PipelineOptions, answer_question
from pipeline_logic import NeedleError, contextual_passage, publish_allowed

log = logging.getLogger("needle")


class PublishBlocked(NeedleError):
    """The new index version was built but not published. `overridable` says whether
    publish_override may force it (a too-small golden set) or not (a real recall drop)."""

    def __init__(self, message: str, *, overridable: bool):
        super().__init__(message)
        self.overridable = overridable


def _golden_rows() -> List[Dict[str, Any]]:
    from eval.schema import load_jsonl

    return load_jsonl(os.getenv("NEEDLE_WORKSPACE_GOLDEN", "").strip() or WORKSPACE_GOLDEN)


def available_documents(collection_name: str) -> Dict[str, set]:
    collection = collection_for(collection_name)
    metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
    names = {str((meta or {}).get("document_name") or "") for meta in metadatas} - {""}
    hashes = set(document_hash_map(collection_name).values())
    return {"names": names, "hashes": hashes}


def _golden_hit(collection_name: str, row: Dict[str, Any], k: int, hash_map: Dict[str, str]) -> float:
    from eval.schema import first_match_rank

    options = PipelineOptions(
        rerank_mode="jev_filter" if REFRESH_GATE_USE_JEV else "fused_only",
        allow_retry=False,
        generate=False,
        collect_ranking=True,
    )
    result = answer_question(
        row.get("question") or "",
        history=row.get("history") or [] if REFRESH_GATE_USE_JEV else [],
        options=options,
        collection_name=collection_name,
    )
    ranked = (result.get("retrieval") or {}).get("ranked") or []
    for parent in ranked:
        parent["content_hash"] = hash_map.get(parent.get("document_id") or "", "")
    return 1.0 if first_match_rank(ranked[:k], row.get("expected") or []) is not None else 0.0


def _golden_gate_rows(collection_name: str) -> List[Dict[str, Any]]:
    """Answerable rows the gate can actually score: their documents are in this index.
    In the free (fused) mode follow-ups are skipped, since condensing them needs a model call."""
    from eval.schema import documents_covered

    available = available_documents(collection_name)
    return [
        row
        for row in _golden_rows()
        if row.get("answerable", True)
        and row.get("expected")
        and documents_covered(row, available)
        and (REFRESH_GATE_USE_JEV or not row.get("history"))
    ]


def _golden_recall(old_collection: str, new_collection: str, rows: List[Dict[str, Any]]) -> tuple:
    if not rows:
        return 1.0, 1.0
    k = env_int("EVAL_K", 5)
    old_hashes = document_hash_map(old_collection)
    new_hashes = document_hash_map(new_collection)
    old_scores = [_golden_hit(old_collection, row, k, old_hashes) for row in rows]
    new_scores = [_golden_hit(new_collection, row, k, new_hashes) for row in rows]
    return sum(old_scores) / len(old_scores), sum(new_scores) / len(new_scores)


def rollback_index() -> Optional[Dict[str, Any]]:
    with mutation_lock:
        return store.rollback_active()


def refresh_active_index(
    embed_style: str = "raw",
    chunking: str = "Parent-child",
    content_hashes: Optional[Dict[str, str]] = None,
    *,
    publish_override: bool = False,
) -> Dict[str, Any]:
    """Copy unchanged vectors, re-embed the rest, then publish if the count and recall gates pass."""
    with mutation_lock:
        return _refresh_active_index(
            embed_style,
            chunking,
            content_hashes or {},
            publish_override=publish_override,
        )


def _refresh_active_index(
    embed_style: str,
    chunking: str,
    content_hashes: Dict[str, str],
    *,
    publish_override: bool = False,
) -> Dict[str, Any]:
    result = _build_index_version(embed_style, chunking, content_hashes, publish=True, publish_override=publish_override)
    return index_status() if result.get("published") else result


def _build_index_version(
    embed_style: str,
    chunking: str,
    content_hashes: Dict[str, str],
    *,
    publish: bool,
    publish_override: bool = False,
    embedding_model_id: Optional[str] = None,
    include_document_names: Optional[List[str]] = None,
    force_reembed: bool = False,
) -> Dict[str, Any]:
    model_id = embedding_model_id or configured_embedding_model()
    active = store.ensure_active_version(configured_embedding_model())
    # The registry row can exist before its collection does (fresh install, nothing uploaded yet).
    old = collection_for(active["collection_name"])
    try:
        payload = old.get(include=["documents", "metadatas", "embeddings"])
    except Exception:
        payload = old.get(include=["documents", "metadatas"])
    ids = list(payload.get("ids") or [])
    documents = list(payload.get("documents") or [])
    metadatas = list(payload.get("metadatas") or [])
    raw_embeddings = payload.get("embeddings")
    emb_by_id: Dict[str, Any] = {}
    # Old vectors are only reusable when they came from the same model.
    same_model = model_id == active["embedding_model"]
    if (
        raw_embeddings is not None
        and not force_reembed
        and same_model
    ):
        for chunk_id, vector in zip(ids, list(raw_embeddings)):
            if vector is not None:
                emb_by_id[chunk_id] = vector
    allowed_names = {name for name in (include_document_names or []) if name}
    if allowed_names:
        kept = [
            (chunk_id, document, metadata)
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
            if (metadata or {}).get("document_name") in allowed_names
        ]
        ids = [item[0] for item in kept]
        documents = [item[1] for item in kept]
        metadatas = [item[2] for item in kept]
    version_id = str(uuid.uuid4())
    collection_name = "needle_" + version_id.replace("-", "")[:12]
    style = embed_style if embed_style in {"raw", "contextual"} else "raw"
    store.begin_version(version_id, model_id, collection_name, embed_style=style, chunking=chunking)
    created = False
    try:
        new_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        created = True
        stored_documents: List[str] = []
        stored_metadata: List[Dict[str, Any]] = []
        embed_inputs: List[str] = []
        copied_embeddings: List[Any] = []
        needs_embed: List[int] = []
        previous_style = active.get("embed_style") or "raw"
        previous_chunking = active.get("chunking") or "Parent-child"
        contextual = style == "contextual"
        config_changed = previous_style != style or previous_chunking != chunking
        for index, (chunk_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
            metadata = dict(metadata or {})
            raw = metadata.get("raw_text") or document or ""
            document_hash = content_hashes.get(metadata.get("document_id") or "")
            hash_changed = bool(document_hash and metadata.get("content_hash") and document_hash != metadata.get("content_hash"))
            prior = emb_by_id.get(chunk_id)
            reusable = (
                not force_reembed
                and same_model
                and not config_changed
                and not hash_changed
                and prior is not None
            )
            metadata["raw_text"] = raw
            metadata["embed_style"] = style
            metadata["chunking"] = chunking
            metadata["version_id"] = version_id
            metadata["embedding_model"] = model_id
            if document_hash:
                metadata["content_hash"] = document_hash
            stored_documents.append(raw)
            stored_metadata.append(metadata)
            if reusable:
                copied_embeddings.append(list(prior))
            else:
                copied_embeddings.append(None)
                needs_embed.append(index)
                embed_inputs.append(contextual_passage(metadata.get("header_context") or "", raw, contextual))
        embed_started = time.perf_counter()
        if needs_embed:
            fresh = embed(embed_inputs, model_id)
            for slot, vector in zip(needs_embed, fresh):
                copied_embeddings[slot] = vector
        embed_ms = (time.perf_counter() - embed_started) * 1000
        if ids:
            embeddings = copied_embeddings
            for start in range(0, len(ids), 100):
                new_collection.add(
                    ids=ids[start:start + 100],
                    documents=stored_documents[start:start + 100],
                    metadatas=stored_metadata[start:start + 100],
                    embeddings=embeddings[start:start + 100],
                )
        if new_collection.count() != len(ids):
            raise RuntimeError("Versioned index handoff aborted because the copy count did not match.")
        published = False
        if publish:
            rows = _golden_gate_rows(active["collection_name"])
            if len(rows) < EVAL_MIN_GOLDEN and not publish_override:
                raise PublishBlocked(
                    f"Only {len(rows)} golden questions cover the documents in this index; the recall check "
                    f"needs at least {EVAL_MIN_GOLDEN}. Publish anyway to skip it, or add questions to "
                    "backend/eval/datasets/workspace.jsonl.",
                    overridable=True,
                )
            old_recall, new_recall = _golden_recall(active["collection_name"], collection_name, rows)
            margin = env_float("EVAL_RECALL_DROP", 0.10)
            if not publish_allowed(
                old_recall,
                new_recall,
                margin,
                golden_count=len(rows),
                min_golden=EVAL_MIN_GOLDEN,
                override=publish_override,
            ):
                raise PublishBlocked(
                    f"Refresh blocked because recall@{env_int('EVAL_K', 5)} on {len(rows)} golden questions "
                    f"fell from {old_recall:.2f} to {new_recall:.2f}.",
                    overridable=False,
                )
            store.publish_version(version_id)
            published = True
        return {
            "version_id": version_id,
            "collection_name": collection_name,
            "embed_style": style,
            "chunking": chunking,
            "embedding_model": model_id,
            "published": published,
            "publish_override": bool(publish_override),
            "chunk_count": len(ids),
            "embedded_count": len(needs_embed),
            "corpus_embed_ms": round(embed_ms, 2),
        }
    except Exception:
        store.fail_version(version_id)
        if created:
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                log.warning("Failed to remove unfinished index %s", collection_name, exc_info=True)
        raise
