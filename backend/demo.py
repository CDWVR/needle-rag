"""Public-demo seeding: load the fictional-company sample documents the eval uses.

Runs once at startup in demo mode, in the background, and skips anything already indexed, so a
restart (or a redeploy onto an existing volume) does not duplicate documents.
"""

import hashlib
import logging

from eval.corpus import corpus_files
from ingest import process_document
from workspace import WorkspaceStore

log = logging.getLogger("needle.demo")


def seed_demo_corpus(workspace: WorkspaceStore) -> int:
    added = 0
    for name, payload in corpus_files():
        digest = hashlib.sha256(payload).hexdigest()
        if workspace.find_hash(digest):
            continue
        content_type = "application/pdf" if name.endswith(".pdf") else "text/plain"
        try:
            doc = process_document(payload, name, content_type, content_hash=digest)
        except Exception:
            log.exception("Could not seed %s", name)
            continue
        workspace.set_policy(doc.id, collection="Demo", byte_size=len(payload), stored_name="", content_hash=digest)
        added += 1
    if added:
        workspace.record_run("Demo seed", f"Loaded {added} sample documents", "Success")
        log.info("Seeded %d demo documents", added)
    return added
