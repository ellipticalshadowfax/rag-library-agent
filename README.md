# RAG Library Agent

A local, self-hosted document assistant. Point it at your ebook/PDF library, let it
build a searchable vector index, then ask questions about your collection in a browser
UI. Everything runs on your machine — no cloud, no data leaves your disk.

Books tagged as Fiction/Short Stories in Calibre are treated as fiction: the assistant
will answer questions about their content but won't present it as factual.

> **New here?** Want the full picture of how the embedding model, semantic search,
> multi-hop retrieval, OCR, and the local LLM all fit together? Read
> [**details.md**](details.md) — a step-by-step walkthrough of the entire RAG chain.

## How it works

Your library files (PDF, EPUB, MOBI) get chunked into small passages and converted
into embedding vectors using a local sentence-transformers model. These vectors live
in a ChromaDB index on disk. When you ask a question, the system:

1. Embeds your question and pulls the closest matching chunks from the index.
2. Reranks those chunks with a cross-encoder for precision.
3. Sends the best chunks as context to an LLM (local or remote) to generate an answer.

The LLM doesn't need to know about your library — it just sees the relevant passages
as context. This is retrieval-augmented generation (RAG).

```
Your library ──► chunk + embed ──► vector index ──► retrieve + rerank ──► LLM ──► answer
  (PDF/EPUB)      (local model)     (ChromaDB)      (top chunks)      (any API)
```

An MCP server is also included so you can use the library as a tool from LM Studio
or any MCP-compatible chat client, instead of the web UI.

## Getting started

**Requirements:** Python 3.10+ on Linux. An LLM API if you want to chat (any
OpenAI-compatible endpoint — local LM Studio, llama.cpp, or a cloud provider).

```bash
git clone <repo-url> rag-library-agent
cd rag-library-agent
chmod +x run.sh
./run.sh
```

That single command:
- Creates a Python virtual environment in `.venv/`
- Installs all dependencies (torch CPU wheels, sentence-transformers, ChromaDB, Flask, etc.)
- Downloads the embedding model once (~50 MB)
- Checks if the LLM API is reachable (warns if not — you can still set it up from the UI)
- Starts the web app at **http://localhost:5000**

### Custom options

```bash
RAG_PORT=8080 ./run.sh        # different port
RAG_HOST=0.0.0.0 ./run.sh     # listen on all interfaces (LAN access)
RAG_DEVICE=gpu ./run.sh        # GPU install (NVIDIA driver + VRAM needed)
```

### Manual setup (if run.sh doesn't work)

```bash
python3 -m venv .venv
# CPU torch first — prevents uv from pulling CUDA wheels
.venv/bin/pip install uv
uv pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt --extra-index-url https://pypi.org/simple
.venv/bin/python scripts/server.py
```

## Using the app

Open http://localhost:5000. The app has four tabs:

**Setup** — Configure your library folders, LLM API endpoint, and embedding model.
There's a folder picker for convenience, and a first-run wizard on fresh installs.
Saving persists everything to `config.json`.

**Scan** — A dry run. Counts your files, flags scanned PDFs that need OCR, and
classifies books as fiction or non-fiction from Calibre tags. Nothing gets indexed.

**Ingest** — Builds or updates the vector index. Runs in the background with live
progress shown on any tab. You can pause, resume, or stop a running job. While
ingesting, the Chat tab is temporarily locked (to avoid concurrent database access).
Re-running is incremental — new files are added, existing ones are skipped.

**Chat** — Ask questions. Each answer shows its sources with similarity scores and
fiction/non-fiction tags. If all your sources are fiction, you get a visible warning.

### Browsing, summaries & quizzes

Beyond plain question-answering, the assistant understands a few intent-shifted
requests that bypass the passage-retrieval cap:

- **"list books about X" / "how many books are in my library"** — a deterministic
  catalog mode renders a Markdown table of matching titles (no LLM involved). If
  more than `catalog_clarify_threshold` titles match, it shows the first 20 and
  answers **"all"** to expand to the full list.
- **"summarize [Book]"** — with `summary_strategy: "map_reduce"`, naming a work
  triggers a deep, book-wide summary (map sections → one streamed reduce call)
  instead of a single shallow pass.
- **"quiz me on [topic/work]"** — generates structured practice questions with
  collapsible answers and per-question source citations. For the full quiz
  pipeline (planning, generation audit, grading analytics, spaced repetition,
  exports), see **[quiz.md](quiz.md)**.

### Study tab: builder, player, review, course & exports

The **Study** tab turns quiz generation into a full study loop. The entire pipeline
— planning, audited generation, grading, spaced repetition via FSRS, and multiple
export formats — is documented in **[quiz.md](quiz.md)**. Here's a quick overview:

- **Builder** — plan an editable quiz against a whole work, explicit sections,
  or a free topic (optionally weighted toward units you've been missing), pick a
  depth and difficulty, then generate an audited quiz (grounded in the work's
  sections, answer-key verified, de-duplicated).
- **Player** — take a stored quiz one question at a time with immediate grading
  (mcq / true-false / fill-blank are deterministic; short answers get optional
  LLM grading) and an end-of-attempt breakdown of accuracy by unit, difficulty,
  and type, plus provenance-linked study points.
- **Review** — missed/weak questions enroll into a shared FSRS spaced-repetition
  store (`quizzes/review.json`) and resurface on due dates as flashcard-style
  cards (Again / Hard / Good / Easy), with retention and strength-over-time.
- **Course** — browse a work's unit syllabus, read a whole-work (or per-unit)
  reading summary, and jump straight to generating a quiz for a single unit.
- **Exports** — any generated quiz exports to Moodle GIFT, CSV, JSON, or an Anki
  `.apkg` deck (`GET /api/quiz/<id>/export?format=...`).

## Configuration

All settings live in `config.json`:

| Key | Default | What it does |
|-----|---------|--------------|
| `embed_model` | `intfloat/multilingual-e5-small` | Sentence-Transformers embedding model |
| `embed_device` | `cpu` | Device to run embeddings on |
| `chunk_tokens` | `330` | Tokens per chunk (keep under model max) |
| `chunk_overlap` | `60` | Overlap between consecutive chunks |
| `llm_base_url` | `http://localhost:1234/v1` | OpenAI-compatible LLM endpoint |
| `llm_model` | `default` | Model ID to request |
| `llm_api_key` | *(empty)* | API key for remote/cloud providers |
| `retrieval_top_k` | `10` | Chunks retrieved per question |
| `ocr_enabled` | `false` | OCR scanned PDFs during ingest |
| `sets` | — | Named library directories (see below) |

The catalog, summary, and quiz behaviors are tuned by additional keys
(`catalog_enabled`, `catalog_clarify_threshold`, `catalog_max_rows`,
`catalog_semantic_pool`, `summary_strategy`, `summary_max_chunks`,
`summary_context_budget`, `quiz_default_count`, `chat_mode`, `show_thinking`,
`max_tokens_*`) — see [**tuning.md**](tuning.md) for how they interact.

> **Tuning & portability:** the defaults work well out of the box, but several
> settings are dataset- or hardware-dependent. See [**tuning.md**](tuning.md) for
> how the RAG parameters behave across different corpora and machines, and how to
> re-tune them (including the hard-coded context budget).

### Library sets

You can organize multiple directories under named "sets":

```json
"sets": {
  "fiction": { "path": "/path/to/fiction/library", "kind": "local" },
  "reference": { "path": "/path/to/reference", "kind": "local" }
}
```

### LLM API

The chat backend talks to any **OpenAI-compatible** endpoint. In the Setup tab you
can set the URL, model, and optional API key, then test the connection. This works
with local servers (LM Studio, llama.cpp, vLLM) or cloud APIs.

## MCP server (for LM Studio)

If you prefer chatting in LM Studio's GUI, the library can be exposed as an MCP
server. The model calls `search_library`, `summarize_work`, `list_collections`,
`list_books`, or `make_quiz` tools while answering questions, grounding its
responses in your actual books.

```bash
./run_mcp.sh              # stdio transport (for local LM Studio)
./run_mcp.sh --http       # HTTP transport (for remote/LAN access, port 8765)
```

Then in LM Studio: Settings → MCP Servers → Add (Local or Remote) and point it
at the script/command.

## OCR

Scanned PDFs (no text layer) are skipped by default and flagged in the Scan tab.
Enable OCR in the config or the Setup tab. The system supports two engines —
Tesseract and RapidOCR — and an automatic comparison tool picks the better one
for each file:

```bash
.venv/bin/python scripts/ocr_compare.py /path/to/library         # compare + OCR
.venv/bin/python scripts/ocr_compare.py /path/to/library --sample-only  # compare only
```

OCR is always CPU-bound and significantly slower than text extraction.

## CLI tools

For power users who prefer the terminal:

```bash
.venv/bin/python scripts/scan.py  /path/to/library          # dry-run stats
.venv/bin/python scripts/ingest.py /path/to/library --set veracrypt1  # build index
.venv/bin/python scripts/agent.py --set veracrypt1          # chat in terminal
```

## Where things live

| File | What it is |
|------|------------|
| `config.json` | All settings (embed model, LLM endpoint, chunking, etc.) |
| `index/` | ChromaDB vector index |
| `manifest.db` | SQLite database tracking file ingest status |
| `ingest.log` | Full ingest log |
| `conversations/` | Saved chat sessions (JSON) |

To rebuild from scratch (e.g. after changing the embedding model), delete `index/`
and `manifest.db`, then re-run ingest. The Setup tab also has a "Force re-embed"
option per file.

## Comparison with similar RAG systems

A lot of popular tools do "chat with your documents." Here's how this project
sits alongside several well-known ones:

| Feature | **RAG Library Agent** | **privateGPT / LocalGPT** | **AnythingLLM** | **LlamaIndex / LangChain** | **Verba (Weaviate)** | **RAGFlow** |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Fully local / no cloud | ✅ | ✅ | ✅ (self-host) | ✅ | ✅ | ✅ |
| Purpose-built for large ebook/PDF libraries | ✅ | ⚠️ | ⚠️ | ❌ framework | ⚠️ | ⚠️ |
| One-command setup | ✅ | ⚠️ | ✅ | ❌ | ⚠️ | ⚠️ |
| Browser UI included | ✅ | ⚠️ (extras) | ✅ | ❌ | ✅ | ✅ |
| Works with *any* OpenAI-compatible LLM (LM Studio, llama.cpp, cloud) | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | ⚠️ |
| Built-in OCR for scanned PDFs | ✅ | ❌ | ⚠️ | ⚠️ (opt-in) | ❌ | ✅ |
| Agentic tool-calling retrieval loop | ✅ | ❌ | ⚠️ | ✅ | ❌ | ⚠️ |
| Hybrid retrieval (dense + BM25) + cross-encoder rerank | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | ✅ |
| Fiction/non-fiction awareness | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multiple named library "sets" | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ | ⚠️ |
| MCP server for external clients (LM Studio) | ✅ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| Offline retrieval eval + CI regression gate | ✅ | ❌ | ❌ | ⚠️ | ❌ | ❌ |
| Lightweight (small dependency footprint, CPU-first) | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ | ❌ |

> Legend: ✅ built-in, ⚠️ possible but requires extra work/configuration, ❌ not provided.

### Remarks on this setup

- **Sized for book-scale libraries, not toy demos.** Most RAG frameworks target
  a handful of documents. This project is built around the reality of indexing
  tens of thousands of chunks across large ebook/PDF collections — deterministic
  chunk IDs for safe re-ingest, batched upserts past Chroma's hard limits, and a
  lazy BM25 index that tolerates ~186k chunks.
- **Everything is self-contained and portable.** One `./run.sh` pulls in the
  venv, dependencies, and embedding model, and the whole folder can be copied to
  another machine. There is no external vector database or framework runtime to
  stand up — just Python and ChromaDB on disk.
- **Local-first by default.** Embeddings run on CPU, and the LLM can be your own
  LM Studio / llama.cpp instance. Your data never has to leave the machine, which
  matters for a private book collection.
- **Opinionated retrieval, not a blank toolkit.** You get hybrid dense+BM25
  fusion, cross-encoder reranking, a low-relevance guard, and a fiction
  awareness layer wired in and tuned out of the box — no assembly required.
  Framework-style tools (LlamaIndex/LangChain) leave these decisions to you.
- **Human-centric workflows for the messy real world.** OCR for scanned PDFs with
  automatic engine comparison, incremental ingest you can pause/resume/stop, and
  a folder-picker UI — conveniences aimed at non-developers that generic
  frameworks don't provide.
- **Good enough, verifiably.** The offline eval harness plus a CI regression gate
  (`scripts/eval.py diff`) let you upgrade embedding models or tweak retrieval
  and prove you didn't make answers worse — something most self-hosted chat apps
  lack.

## Notes

- **GPU install is for future use only.** The app currently forces CPU at runtime
  for embeddings regardless of install mode. OCR is also CPU-only.
- **Fiction classification** depends on Calibre tags. Untagged books default to
  non-fiction.
- **Embedding model tradeoffs:** The default (`multilingual-e5-small`) is fast on
  CPU (~10 files/min). Bigger models (`bge-m3`, `Qwen3-Embedding`) give better
  quality but are 4-10x slower.
- **Portable:** Copy the whole folder to another machine and run `./run.sh`. The
  library itself stays wherever it is.
