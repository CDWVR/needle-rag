# 🧭 Needle — Enterprise RAG Platform

> An industrial-grade Retrieval-Augmented Generation (RAG) system featuring Multi-Modal Parsing, Hybrid Search (Vector + Keyword), Reciprocal Rank Fusion, and Parent-Child Chunking. Designed with a premium minimalist aesthetic.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite_FTS5-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)

---

## ✨ Features

- **🌐 Global Database Chat** — Query your entire unified database, or isolate to a single document.
- **⚡ Vector search with a similarity gate** — Embeds the question, retrieves the top children, and drops hits below an absolute cosine floor or a drop-off from the best match. Exact keyword hits are extra candidates, not a substitute for that gate.
- **⚖️ Jev reranking** — [Jev](https://pypi.org/project/jev-reranker/) scores every surviving passage for usefulness as evidence and discards the ones that only share a topic. Parent chunks are loaded after that decision.
- **🏗️ Parent-Child Chunking** — Granular 400-char child chunks for laser-focused semantic retrieval, seamlessly mapped back to massive 2,000-char parent chunks to preserve surrounding context.
- **📊 Metadata Scaffolding** — Automatically extracts Markdown headers (Chapters/Sections) to provide spatial awareness to citations.
- **📄 Multi-Modal Parsing** — Processes text-based PDFs, DOCX, PPTX, and XLSX using `PyMuPDF` and Microsoft's `MarkItDown`.
- **🆓 Unlimited Free Embeddings** — Swapped out cloud embeddings for local, highly-optimized `all-MiniLM-L6-v2` running via ONNX Runtime for zero rate limits and maximum privacy.
- **🚀 Async Ingestion** — FastAPI background tasks ensure massive 400-page textbooks process without blocking the server.
- **🎨 Premium UI/UX** — A handcrafted Charcoal/Zinc aesthetic featuring fluid glassmorphism, responsive floating chat components, and a perfectly centered conversational layout.

---

## 🏗️ Enterprise RAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│     Global Chat  │  Isolated Mode  │  Source Citations  │
└────────┬──────────────────┬───────────────────┬─────────┘
         │                  │                   │
    POST /upload       POST /chat        GET /documents
         │                  │                   │
┌────────▼──────────────────▼───────────────────▼─────────┐
│                    FastAPI BACKEND                      │
│                                                         │
│  [1] MULTI-MODAL PARSER (MarkItDown / PyMuPDF)          │
│        ↓ (Rejects Corrupted / Encrypted files)          │
│  [2] METADATA SCAFFOLDING (Header Extractor)            │
│        ↓                                                │
│  [3] PARENT-CHILD CHUNKER (2000-char / 400-char)        │
│        ↓                                                │
│  ┌─────┴─────┐                                          │
│  ↓           ↓                                          │
│ CHROMA DB  SQLITE FTS5                                  │
│ (Vectors)  (Keywords)                                   │
│  │           │                                          │
│  └─────┬─────┘                                          │
│        ↓                                                │
│  [4] TOP-K VECTOR SEARCH + SIMILARITY THRESHOLD        │
│        ↓                                                │
│  [5] JEV RERANKER (keep useful evidence only)           │
│        ↓                                                │
│  [6] FETCH PARENT CHUNK → PROMPT + CITATIONS            │
│        ↓                                                │
│  [7] OPENROUTER DRAFT → GROUNDED / SAFE / RELEVANT     │
│        ↓                                                │
│     Cited answer, or a low-confidence no-answer         │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | Async HTTP server with Background Tasks |
| **Error Handling** | Exception Middleware | Graceful rejection of corrupted/encrypted files |
| **LLM Synthesis** | OpenRouter (`deepseek/deepseek-v4.1-flash` writer, `deepseek/deepseek-v4-flash` checker) | Cited draft and the grounding check |
| **Reranker** | Jev `typesafe/jev-1.13` via OpenRouter | Keeps only passages that are useful evidence |
| **Local Embeddings** | `all-MiniLM-L6-v2` | Open-source, CPU-optimized semantic vectors |
| **Vector DB** | ChromaDB | Persistent vector similarity search |
| **Keyword DB** | SQLite FTS5 | Blazing fast parallel sparse indexing (BM25) |
| **Parsers** | PyMuPDF, MarkItDown | Page-aware PDF parsing and multi-modal conversions |
| **Frontend** | Vanilla JS / CSS | Premium dark UI with SSE streaming and Global Chat |

---

## 📖 Deep Dive: How the Engine Works

### 1. Ingestion & Scaffolding
When a document is uploaded, it is passed to either `PyMuPDF` (to preserve exact page numbers) or `MarkItDown` (to parse Word, Excel, PPTX). The `MarkdownHeaderTextSplitter` injects hierarchical scaffolding into the chunk metadata (e.g. `[Chapter 2 > Safety]`).

### 2. Parent-Child Chunking and Index Cards
Text is split into **2,000-character parent chunks**, then into **400-character children**. Each parent also gets a short index card so a heading-level summary can be retrieved. Only those units are embedded. Parent text is stored separately and loaded after Jev decides which hits are worth expanding.

### 3. Retrieval, Jev, and the answer check
The question is tokenized and embedded with the same `all-MiniLM-L6-v2` model recorded on the active index. Search refuses to run when that version is incompatible. Vector hits below cosine similarity `0.30`, or below 70% of the best hit, are dropped. Keyword matches can still be offered to Jev, which then keeps passages at relevance `0.20` or higher. The top parents are assembled with citations. An OpenRouter chat model writes a draft, and a second pass releases it only when it is grounded, safe, and relevant. Otherwise the user gets a no-answer fallback and no citations. Jev itself is called through OpenRouter at `https://openrouter.ai/api/v1/systemone`, so one `OPENROUTER_API_KEY` covers reranking and answers.

Uploads and deletes write the active index. `POST /api/index/refresh` rebuilds a new collection and publishes it only after the copied count matches. Deletes are applied to every non-failed version in the registry.

---

## 🚧 Known Limitations & Roadmap

While the architecture is highly advanced, a true production deployment requires addressing the following edge cases:

1. **Tabular Data (XLSX/CSV) Chunking:** Currently, `MarkItDown` converts spreadsheets to Markdown tables, which are then passed through the standard 400-char parent-child chunker. This risks silently breaking rows mid-chunk for heavy tabular data (e.g., ILMT reports). *Roadmap:* Implement row-wise chunking specifically for spreadsheets to preserve row integrity and column headers.
2. **Scanned Documents (OCR):** `PyMuPDF` is exceptionally fast for digital PDFs but lacks built-in OCR. Scanned signed contracts will extract as empty or garbled text. *Roadmap:* Implement a Tesseract or cloud OCR fallback path for image-based PDFs.
3. **Advanced Error Recovery:** While password-protected or corrupted files are caught and rejected gracefully, partial failures (e.g., extracting 50 out of 100 pages before corruption) should support partial ingestion states.

### ☁️ Hosting Disclaimer (Render Free Tier)
The live demonstration is hosted on Render's free tier, which imposes a strict **512MB RAM limit**. Because the architecture utilizes a completely local, privacy-first embedding model (`all-MiniLM-L6-v2`) rather than calling external APIs for chunk vectorization, the memory overhead during document ingestion often exceeds 512MB. 
**As a result, document uploads on the live hosted version may fail due to Out-Of-Memory (OOM) crashes.** For a stable, flawless experience, please run the application locally following the Quick Start guide below.
---

## 🚀 Quick Start

### Prerequisites

- Python 3.13
- OpenRouter API key ([openrouter.ai/keys](https://openrouter.ai/keys)), used for both [Jev](https://openrouter.ai/typesafe/jev-1.13) and the answer model

### Setup

```bash
# 1. Unzip and navigate to the extracted project folder
cd <extracted-folder-name>

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Set your API key
copy .env.example .env
# Edit .env and add OPENROUTER_API_KEY

# 5. Run the server
cd backend
python main.py
```

### Open in Browser
Navigate to **http://localhost:8000**

---

## 📝 License

MIT — built for educational purposes and enterprise architectural demonstrations.
