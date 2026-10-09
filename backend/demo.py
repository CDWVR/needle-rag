"""Public-demo seeding: load the fictional-company sample documents the eval uses.

Runs once at startup in demo mode, in the background, and skips anything already indexed, so a
restart (or a redeploy onto an existing volume) does not duplicate documents.
"""

import hashlib
import logging

from catalog import delete_document, get_all_documents
from eval.corpus import corpus_files, stable_payload
from ingest import process_document
from workspace import WorkspaceStore

log = logging.getLogger("needle.demo")
COLLECTION = "Demo"


def _prune_duplicates(workspace: WorkspaceStore, digests: dict) -> int:
    """Keep one copy of each sample document. Earlier releases hashed the rendered handbook PDF, whose
    bytes change on every run, so each restart added another copy."""
    policies = workspace.policies()
    by_name: dict = {}
    for doc in get_all_documents():
        if (policies.get(doc["id"]) or {}).get("collection") == COLLECTION and doc["name"] in digests:
            by_name.setdefault(doc["name"], []).append(doc["id"])
    removed = 0
    for name, ids in by_name.items():
        keep = next((i for i in ids if policies[i].get("content_hash") == digests[name]), ids[0])
        if policies[keep].get("content_hash") != digests[name]:
            workspace.set_policy(keep, collection=COLLECTION, byte_size=policies[keep].get("bytes") or 0,
                                 stored_name="", content_hash=digests[name])
        for document_id in ids:
            if document_id != keep:
                workspace.tombstone(document_id)
                delete_document(document_id)
                workspace.forget_document(document_id)
                removed += 1
    return removed


def seed_demo_corpus(workspace: WorkspaceStore) -> int:
    files = corpus_files()
    digests = {name: hashlib.sha256(stable_payload(name, payload)).hexdigest() for name, payload in files}
    removed = _prune_duplicates(workspace, digests)
    if removed:
        log.info("Removed %d duplicate demo documents", removed)
    added = 0
    for name, payload in files:
        digest = digests[name]
        if workspace.find_hash(digest):
            continue
        content_type = "application/pdf" if name.endswith(".pdf") else "text/plain"
        try:
            doc = process_document(payload, name, content_type, content_hash=digest)
        except Exception:
            log.exception("Could not seed %s", name)
            continue
        workspace.set_policy(doc.id, collection=COLLECTION, byte_size=len(payload), stored_name="", content_hash=digest)
        added += 1
    if added:
        workspace.record_run("Demo seed", f"Loaded {added} sample documents", "Success")
        log.info("Seeded %d demo documents", added)
    return added
