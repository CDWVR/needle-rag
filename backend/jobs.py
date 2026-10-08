"""Background ingestion. An upload request validates the file, queues it, and returns at once;
one worker thread parses and embeds queued files in order (writes to the index are serialised
anyway), and the browser polls the job until it finishes.
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Set

from ingest import process_document
from pipeline_logic import NeedleError
from workspace import WorkspaceStore

log = logging.getLogger("needle.jobs")


class IngestionQueue:
    def __init__(self, workspace: WorkspaceStore, upload_dir: str):
        self.workspace = workspace
        self.upload_dir = upload_dir
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._lock = threading.Lock()
        # Hashes queued but not yet indexed, so a double click does not index a file twice.
        self._pending: Set[str] = set()

    def claim(self, content_hash: str) -> bool:
        """Reserve a content hash; False if the same bytes are already queued."""
        with self._lock:
            if content_hash in self._pending:
                return False
            self._pending.add(content_hash)
            return True

    def submit(self, *, job_id: str, data: bytes, filename: str, ext: str, content_hash: str, collection: str) -> None:
        self._pool.submit(self._run, job_id, data, filename, ext, content_hash, collection)

    def _run(self, job_id: str, data: bytes, filename: str, ext: str, content_hash: str, collection: str) -> None:
        self.workspace.update_job(job_id, "processing")
        try:
            content_type = "application/pdf" if ext == ".pdf" else ""
            doc = process_document(data, filename, content_type, content_hash=content_hash)
        except Exception as exc:
            if not isinstance(exc, (ValueError, NeedleError)):
                log.exception("Ingestion failed for job %s", job_id)
            message = str(exc) if isinstance(exc, (ValueError, NeedleError)) else "The file could not be indexed."
            self.workspace.update_job(job_id, "failed", message)
            self.workspace.record_run("Ingestion", f"{filename} failed", "Failed")
            return
        finally:
            with self._lock:
                self._pending.discard(content_hash)
        stored_name = f"{doc.id}{ext}"
        path = os.path.join(self.upload_dir, stored_name)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        self.workspace.set_policy(
            doc.id, collection=collection, byte_size=len(data), stored_name=stored_name, content_hash=content_hash
        )
        self.workspace.update_job(job_id, "completed", document_id=doc.id)
        self.workspace.record_run("Ingestion", f"{filename}: {doc.num_chunks} chunks", "Success")

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def job(self, job_id: str) -> Optional[dict]:
        return self.workspace.job(job_id)
