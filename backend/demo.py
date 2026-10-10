"""Public-demo seeding: load the machine-learning sample documents in backend/demo_corpus/.

The eval keeps its own fictional-company corpus (eval/corpus/); the two are deliberately separate, so
changing what the demo shows never moves the eval baselines.

Runs once at startup in demo mode, in the background, and skips anything already indexed, so a
restart (or a redeploy onto an existing volume) does not duplicate documents.
"""

import hashlib
import logging
import os

from catalog import delete_document, get_all_documents
from eval.corpus import corpus_files, stable_payload
from ingest import process_document
from workspace import WorkspaceStore

log = logging.getLogger("needle.demo")
COLLECTION = "Demo"
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_corpus")


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


def _prune_stale(workspace: WorkspaceStore, names: set) -> int:
    """Remove demo documents that are no longer part of the demo set (an earlier release shipped a
    different one, and a redeploy keeps the volume)."""
    policies = workspace.policies()
    removed = 0
    for doc in get_all_documents():
        if (policies.get(doc["id"]) or {}).get("collection") == COLLECTION and doc["name"] not in names:
            workspace.tombstone(doc["id"])
            delete_document(doc["id"])
            workspace.forget_document(doc["id"])
            removed += 1
    return removed


def seed_demo_corpus(workspace: WorkspaceStore) -> int:
    files = corpus_files(DEMO_DIR)
    digests = {name: hashlib.sha256(stable_payload(name, payload, DEMO_DIR)).hexdigest() for name, payload in files}
    stale = _prune_stale(workspace, {name for name, _ in files})
    if stale:
        log.info("Removed %d documents from an earlier demo set", stale)
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
