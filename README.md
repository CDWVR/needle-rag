# 🧭 Needle — Enterprise RAG Platform

> An industrial-grade Retrieval-Augmented Generation (RAG) system featuring Multi-Modal Parsing, Hybrid Search (Vector + Keyword), Reciprocal Rank Fusion, and Parent-Child Chunking. Designed with a premium minimalist aesthetic.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite_FTS5-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-6566F1?style=for-the-badge)

---

## ✨ Features

- **🌐 Global Database Chat** — Query your entire unified database, or isolate to a single document.
- **⚡ Hybrid search** — Vector hits above a cosine floor are fused (RRF) with BM25 keyword hits and, when the question names a code such as `E-104` or `TERN-SCALE`, an exact-phrase match list.
- **⚖️ Jev reranking** — [Jev](https://pypi.org/project/jev-reranker/) scores every surviving passage for usefulness as evidence and discards the ones that only share a topic. Parent chunks are loaded after that decision.
- **🏗️ Parent-Child Chunking** — Granular 400-char child chunks for laser-focused semantic retrieval, seamlessly mapped back to massive 2,000-char parent chunks to preserve surrounding context.
- **📊 Metadata Scaffolding** — Automatically extracts Markdown headers (Chapters/Sections) to provide spatial awareness to citations.
- **📄 Multi-Modal Parsing** — Processes text-based PDFs, DOCX, PPTX, and XLSX using `PyMuPDF` and Microsoft's `MarkItDown`.
- **🆓 Unlimited Free Embeddings** — Swapped out cloud embeddings for local, highly-optimized `all-MiniLM-L6-v2` running via ONNX Runtime for zero rate limits and maximum privacy.
- **🚀 Non-blocking Ingestion** — Uploads are parsed and embedded on a worker thread, so a 400-page textbook does not stall chat or other requests.
- **🧪 Production eval** — A repo-owned test corpus and golden set run through the exact production pipeline, gated in CI. See [Evaluation](#-evaluation).
- **🎨 Workspace UI** — Ask, knowledge base, document detail, pipeline, analytics, and settings views with retrieval traces and source inspection. Light and dark themes (System / Light / Dark in the account menu or the Ctrl K command menu), page transitions, and motion that respects the reduced-motion setting.

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
| **Backend** | FastAPI | HTTP API with a background ingestion queue |
| **Security** | Token sign-in, signed sessions, CSP | See [SECURITY.md](SECURITY.md) |
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
The question is tokenized and embedded with the same `all-MiniLM-L6-v2` model recorded on the active index. Search refuses to run when that version is incompatible. Vector hits below cosine similarity `0.30` are dropped. Keyword matches can still be offered to Jev, which then keeps passages at relevance `0.20` or higher. The top parents are assembled with citations. An OpenRouter chat model writes a draft, and a second pass releases it only when it is grounded, safe, and relevant. Otherwise the user gets a no-answer fallback and no citations. Jev itself is called through OpenRouter at `https://openrouter.ai/api/v1/systemone`, so one `OPENROUTER_API_KEY` covers reranking and answers.

Chat and the eval run the same code path (`run_pipeline` in `backend/pipeline.py`), so evaluation measures what users get.

Uploads and deletes write the active index. `POST /api/index/refresh` rebuilds a new collection and publishes it only after the copied count matches and recall@5 on your workspace golden questions has not dropped by more than 0.10 (fused ranking, so the check costs nothing). If fewer than 30 golden questions cover your documents, the UI offers to publish without that comparison. Deletes are applied to every non-failed version in the registry.

### Jev configuration

Jev is called through OpenRouter, so `OPENROUTER_API_KEY` covers reranking and answers. It scores the top 15 fused candidates (`JEV_CANDIDATE_LIMIT`); on both eval corpora the evidence is always inside that window. Timeouts (`JEV_TIMEOUT_SECONDS=30`) and retries (`JEV_MAX_RETRIES=2`) are kept short so a slow Jev falls back to a local ranker instead of stalling a chat. Every option is listed in `.env.example`.

---

### Public demo

`NEEDLE_DEMO_MODE=true` serves a read-only demo with sample documents; visitors can ask questions
but cannot change anything. A `Dockerfile` and `railway.toml` are included; see
[docs/deploy-railway.md](docs/deploy-railway.md).

### Code map (`backend/`)

| Module | Responsibility |
| --- | --- |
| `main.py` | HTTP routes and middleware wiring |
| `security.py` | Sign-in, sessions, CSRF, security headers, rate limits, upload validation |
| `jobs.py` | Background ingestion queue |
| `demo.py` | Public-demo sample-corpus seeding |
| `config.py` | Every environment setting, read once |
| `ingest.py` | Parse, chunk, embed, and store an upload |
| `pipeline.py` | The question path: condense, hybrid search, Jev, write, check |
| `pipeline_logic.py` | Pure gates and scoring rules (unit-tested, no I/O) |
| `rerank.py`, `llm.py`, `embedder.py` | Jev, OpenRouter chat, and embedding clients |
| `catalog.py`, `index_store.py` | Chroma collections, parent/keyword store, document catalog |
| `versions.py` | Index versions, the refresh recall gate, rollback |
| `workspace.py` | Settings, conversations, jobs, analytics |
| `eval/` | The evaluation harness (see below) |

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
# First start prints an access token (also saved in backend/secrets/access_token).
# Or set NEEDLE_ACCESS_TOKEN in .env before starting.
```

### Open in Browser
Navigate to **http://localhost:8000** and sign in with the access token.

---

## 🧪 Evaluation

The eval lives in `backend/eval/` and is run from `backend/`:

```bash
python -m eval validate
```

```bash
python -m eval run --suite hermetic --tier offline
```

```bash
python -m eval run --suite hermetic --tier full --max-cost 0.50
```

**Suites**
- `hermetic` — `eval/corpus/` holds eleven documents about a fictional company (Markdown, text, CSV, and a PDF rendered at run time so page citations are tested), including two conflicting policy versions and a passage carrying an injected instruction. `eval/datasets/hermetic.jsonl` has 117 questions: factual, exact-code, multi-hop, follow-up, near-miss unanswerable, off-topic, and injection cases. Every run ingests the corpus through the real upload code into a throwaway data directory, so it never touches your workspace.
- `demo` — the public demo's four documents in `backend/demo_corpus/` (RNN and retrieval notes, plus two lecture decks on attention and autoencoders, included for educational purposes). `eval/datasets/demo.jsonl` has 65 questions, including the sample questions shown on the demo's home screen, built the same way as `hermetic`. CI runs its offline tier too.
- `workspace` — `eval/datasets/workspace.jsonl` asks about your own documents and runs read-only against your live index. Rows whose documents are not indexed are skipped. Grow it with `python -m eval draft` then `python -m eval review`.

**Tiers**
- `offline` — retrieval only, fused ranking, no model calls: free and deterministic. Reports hit@1/3/5/10, recall@k, MRR, nDCG, and page-level hits. CI runs it on every push.
- `full` — the production path (condense, Jev, retry, writer, checker). Adds answer rate, false-refusal rate, key-fact recall, citation validity, whether citations point at the expected evidence, abstention precision/recall on unanswerable questions, injection leaks, per-stage latency, and cost. `--judge` adds an LLM correctness grade.

**Gates** — `eval/config.json` sets absolute floors per suite and tier, plus the largest allowed drop against the committed baseline in `eval/baselines/`. Model-backed tiers compare question by question with a bootstrap confidence interval, so noise on a few questions does not fail the gate. Exit codes: 0 pass, 1 gate failed, 2 dataset or setup error, 3 stopped by budget or infrastructure errors. `--update-baseline` records a new baseline. Reports go to `eval/reports/<suite>/` (JSON, Markdown, and per-question JSONL), and the Pipeline and Analytics views show the latest result.

---

## 📝 License

MIT — built for educational purposes and enterprise architectural demonstrations.
