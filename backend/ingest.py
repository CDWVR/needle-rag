"""Parse an uploaded file, chunk it parent-child, embed the units, and write them to the active index."""

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from catalog import (
    collection_for,
    mutation_lock,
    require_compatible,
    store,
)
from embedder import embed
from pipeline_logic import contextual_passage, index_card, tokenize

log = logging.getLogger("needle")


@dataclass
class DocumentInfo:
    id: str
    name: str
    num_chunks: int
    num_pages: int
    file_type: str
    uploaded_at: str


HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)


def header_context(metadata: Dict[str, Any]) -> str:
    parts = [
        metadata.get("Header 1", ""),
        metadata.get("Header 2", ""),
        metadata.get("Header 3", ""),
    ]
    return " > ".join(part for part in parts if part)


def _ocr_pdf(path: str) -> str:
    if os.getenv("OCR_ENABLED", "false").strip().lower() not in {"1", "true", "yes"}:
        return ""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        log.warning("OCR is enabled but pytesseract or pdf2image is not installed")
        return ""
    try:
        images = convert_from_path(path)
    except Exception:
        log.exception("OCR could not read %s", path)
        return ""
    pages = []
    for index, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image) or ""
        if text.strip():
            pages.append(f"# Page {index}\n{text.strip()}")
    return "\n\n".join(pages)


def process_document(file_bytes: bytes, filename: str, file_type: str, chunking: str = "Parent-child", content_hash: str = "") -> DocumentInfo:
    active = require_compatible()
    chunking = active.get("chunking") or chunking or "Parent-child"
    document_id = str(uuid.uuid4())
    suffix = os.path.splitext(filename)[1] or ".bin"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(file_bytes)
            temp_path = handle.name

        if file_type == "application/pdf" or filename.lower().endswith(".pdf"):
            raw_docs = PyMuPDFLoader(temp_path).load()
            ftype = "pdf"
            num_pages = len(raw_docs)
            if num_pages == 0 and not raw_docs:
                raise ValueError("No text could be extracted.")
            md_splits = []
            for doc in raw_docs:
                page_text = doc.page_content.strip()
                if not page_text:
                    continue
                for split in md_splitter.split_text(page_text):
                    split.metadata["page"] = doc.metadata.get("page", 0) + 1
                    md_splits.append(split)
        else:
            converted = MarkItDown().convert(temp_path)
            full_text = converted.text_content or ""
            if not full_text.strip():
                raise ValueError("No text could be extracted.")
            ftype = "multimodal"
            num_pages = 1
            md_splits = md_splitter.split_text(full_text)
            for split in md_splits:
                split.metadata["page"] = 1

        if not md_splits and (file_type == "application/pdf" or filename.lower().endswith(".pdf")):
            ocr_text = _ocr_pdf(temp_path)
            if ocr_text.strip():
                md_splits = md_splitter.split_text(ocr_text)
                for split in md_splits:
                    split.metadata["page"] = 1
                ftype = "pdf"
                num_pages = max(num_pages, 1)
        if not md_splits:
            raise ValueError("No text could be extracted.")

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        parent_chunks = parent_splitter.split_documents(md_splits)
        if not parent_chunks:
            raise ValueError("No meaningful text chunks could be created.")

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        parent_rows: List[Dict[str, Any]] = []
        fts_rows: List[tuple] = []

        for parent_index, parent in enumerate(parent_chunks):
            header = header_context(parent.metadata)
            page_number = int(parent.metadata.get("page", 1) or 1)
            parent_key = f"{document_id}_{parent_index}"
            summary = index_card(header, parent.page_content)
            parent_rows.append(
                {
                    "parent_id": parent_key,
                    "page_number": page_number,
                    "chunk_index": parent_index,
                    "header_context": header,
                    "parent_text": parent.page_content,
                    "summary": summary,
                }
            )
            children = child_splitter.split_text(parent.page_content)
            if chunking == "Index card summary":
                card = summary or parent.page_content[:CHILD_CHUNK_SIZE]
                units = [("index_card", 0, card)]
            elif chunking == "Fixed window":
                units = [("child", child_index, child_text) for child_index, child_text in enumerate(children)]
            else:
                units = [("child", child_index, child_text) for child_index, child_text in enumerate(children)]
                if summary and all(summary != child_text for _, _, child_text in units):
                    units.append(("index_card", len(children), summary))

            for unit_type, child_index, passage in units:
                chunk_id = (
                    f"{document_id}_{parent_index}_card"
                    if unit_type == "index_card"
                    else f"{document_id}_{parent_index}_{child_index}"
                )
                ids.append(chunk_id)
                texts.append(passage)
                metadatas.append(
                    {
                        "document_id": document_id,
                        "document_name": filename,
                        "page_number": page_number,
                        "chunk_index": parent_index,
                        "header_context": header,
                        "parent_id": parent_key,
                        "unit_type": unit_type,
                        "token_count": len(tokenize(passage)),
                        "version_id": active["version_id"],
                        "chunking": chunking,
                        "content_hash": content_hash,
                    }
                )
                fts_rows.append((document_id, chunk_id, filename, page_number, header, passage))

        uploaded_at = datetime.now(timezone.utc).isoformat()
        with mutation_lock:
            current = require_compatible()
            for metadata in metadatas:
                metadata["version_id"] = current["version_id"]
            collection = collection_for(current["collection_name"])
            use_context = (current.get("embed_style") or "raw") == "contextual"
            for metadata, passage in zip(metadatas, texts):
                metadata["raw_text"] = passage
                metadata["embed_style"] = "contextual" if use_context else "raw"
            embeddings = embed([
                contextual_passage(metadata.get("header_context") or "", passage, use_context)
                for metadata, passage in zip(metadatas, texts)
            ])
            for start in range(0, len(ids), 100):
                collection.add(
                    ids=ids[start:start + 100],
                    documents=texts[start:start + 100],
                    metadatas=metadatas[start:start + 100],
                    embeddings=embeddings[start:start + 100],
                )
            store.save_document(
                document_id=document_id,
                name=filename,
                num_pages=num_pages,
                num_chunks=len(ids),
                file_type=ftype,
                uploaded_at=uploaded_at,
                version_id=current["version_id"],
                parents=parent_rows,
                fts_rows=fts_rows,
            )
        return DocumentInfo(
            id=document_id,
            name=filename,
            num_chunks=len(ids),
            num_pages=num_pages,
            file_type=ftype,
            uploaded_at=uploaded_at,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
