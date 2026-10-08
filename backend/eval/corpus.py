"""The hermetic eval corpus: documents that ship with the repo.

Files in eval/corpus/ are ingested as-is, except `*.pdf.txt`, which is a text
source rendered to a real PDF (pages split on lines containing only `\\f`) so the
PDF parser and page-number citations are exercised too.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Dict, List, Tuple

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
PAGE_BREAK = "\\f"


def _render_pdf(source: str) -> bytes:
    try:
        import pymupdf as fitz  # PyMuPDF, already required by the PDF parser
    except ImportError:  # older PyMuPDF releases only ship the `fitz` name
        import fitz

    with open(source, encoding="utf-8") as handle:
        text = handle.read().replace("\r\n", "\n")
    pages = [page.strip("\n") for page in text.split(f"\n{PAGE_BREAK}\n")]
    document = fitz.open()
    for page_text in pages:
        page = document.new_page(width=595, height=842)  # A4 in points
        overflow = page.insert_textbox(fitz.Rect(56, 56, 539, 786), page_text, fontsize=10.5, fontname="helv")
        if overflow < 0:
            raise ValueError(f"{os.path.basename(source)}: a page overflows the A4 text box; split it with {PAGE_BREAK}")
    document.set_metadata({"title": os.path.basename(source)[: -len(".txt")], "creationDate": "D:20260101000000Z"})
    return document.tobytes(deflate=True, garbage=3)


def corpus_files() -> List[Tuple[str, bytes]]:
    """(file name as uploaded, bytes) for every corpus document, in a stable order."""
    files = []
    for name in sorted(os.listdir(CORPUS_DIR)):
        path = os.path.join(CORPUS_DIR, name)
        if not os.path.isfile(path) or name.startswith((".", "_")):
            continue
        if name.endswith(".pdf.txt"):
            files.append((name[: -len(".txt")], _render_pdf(path)))
        else:
            with open(path, "rb") as handle:
                # Same bytes on every platform, so chunk boundaries (and metrics) match CI.
                files.append((name, handle.read().replace(b"\r\n", b"\n")))
    return files


def corpus_fingerprint() -> str:
    digest = hashlib.sha256()
    for name, payload in corpus_files():
        digest.update(name.encode())
        digest.update(hashlib.sha256(_stable(name, payload).replace(b"\r\n", b"\n")).digest())
    return digest.hexdigest()[:16]


def _stable(name: str, payload: bytes) -> bytes:
    # Rendered PDFs embed object ids that can differ between PyMuPDF builds; fingerprint the source instead.
    if name.endswith(".pdf"):
        with open(os.path.join(CORPUS_DIR, name + ".txt"), "rb") as handle:
            return handle.read()
    return payload


def build_index(data_dir: str) -> Dict[str, Dict]:
    """Ingest the corpus through the production upload path into an isolated data dir.

    NEEDLE_DATA_DIR must already point at `data_dir` before the engine is imported.
    Returns {file name: {"document_id", "chunks", "pages"}}.
    """
    from paths import DATA_DIR

    if os.path.abspath(DATA_DIR) != os.path.abspath(data_dir):
        raise RuntimeError("Set NEEDLE_DATA_DIR before importing the engine; refusing to touch another index.")
    from catalog import get_all_documents
    from ingest import process_document

    if get_all_documents():
        raise RuntimeError(f"{data_dir} already has documents; the hermetic suite needs an empty data dir.")
    built = {}
    for name, payload in corpus_files():
        content_type = "application/pdf" if name.endswith(".pdf") else "text/plain"
        info = process_document(payload, name, content_type, content_hash=hashlib.sha256(payload).hexdigest())
        built[name] = {"document_id": info.id, "chunks": info.num_chunks, "pages": info.num_pages}
    return built


def fresh_data_dir(prefix: str = "needle-eval-") -> str:
    return tempfile.mkdtemp(prefix=prefix)
