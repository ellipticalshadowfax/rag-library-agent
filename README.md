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

1. Retrieves candidates via **hybrid search** — both dense (embedding) and sparse
   (BM25 lexical) indexes fused by Reciprocal Rank Fusion, then re-ranked by a
   cross-encoder for precision.
2. Checks each candidate against a low-relevance guard (proper-noun presence check)
   to avoid wasting turns on irrelevant matches.
3. Sends the best chunks as context to an LLM (local or remote) to generate an answer.

The LLM doesn't need to know about your library — it just sees the relevant passages
as context. This is retrieval-augmented generation (RAG).

For chat, the agent supports **agentic tool-calling** (`chat_mode: "agentic"`): the LLM can call
`search_library`, `summarize_work`, `list_books`, or `make_quiz` mid-conversation, letting it
self-direct retrieval rather than accepting one pass. For models without function-calling,
it falls back to a ReAct text protocol automatically.
```
Your library ──► chunk + embed ──► vector index ──► retrieve + rerank ──► LLM ──► answer
  (PDF/EPUB)      (local model)     (ChromaDB)      (dense + BM25)      (any API)
```

An MCP server is also included so you can use the library as a tool from LM Studio
or any MCP-compatible chat client, instead of the web UI. MCP exposes tools for
searching, summarizing, listing, quiz planning & building, grading, review queues,
and even config editing — all share the same persistence layer as the web app.

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
- Installs the **core** dependencies from `pyproject.toml` (torch CPU wheels, sentence-transformers, ChromaDB, Flask, MCP, etc.). Installing extras pulls in optional heavy deps only if you ask for them.
- Downloads the embedding model once (~50 MB)
- Checks if the LLM API is reachable (warns if not — you can still set it up from the UI)
- Starts the web app at **http://localhost:5000**

### Custom options

```bash
RAG_PORT=8080 ./run.sh                        # different port
RAG_HOST=0.0.0.0 ./run.sh                     # listen on all interfaces (LAN access)
RAG_DEVICE=gpu ./run.sh                       # GPU install (NVIDIA driver + VRAM needed)
RAG_PIP_EXTRAS=rapidocr ./run.sh              # also install the optional rapidocr OCR backend
RAG_PIP_EXTRAS=rapidocr,anki ./run.sh         # plus Anki .apkg quiz export
```

Optional extras (installed via `RAG_PIP_EXTRAS` or `pip install ".[extra]"`):

| Extra | Adds | Why it's optional |
|-------|------|-------------------|
| (none — default OCR) | `pytesseract` (system `tesseract` binary) | Default; light & fast |
| `rapidocr` | `rapidocr-onnxruntime`, `onnxruntime`, `onnx`, `opencv-python` | Heavy (~0.5–1 GB) factorized OCR backend; install only for higher-accuracy scanned-PDF OCR |
| `anki` | `genanki` | Anki `.apkg` deck export |
| `gpu` | CUDA torch + NVIDIA runtime libs | Requires a CUDA-capable GPU; resolved by run.sh via the PyTorch index |

### Manual setup (if run.sh doesn't work)

```bash
python3 -m venv .venv
.venv/bin/pip install uv
# CPU-only torch first — PyPI's default torch bundles CUDA on Linux (~870MB+);
# the +cpu wheel is ~200MB. (For GPU, use RAG_DEVICE=gpu ./run.sh instead.)
uv pip install "torch" --index-url https://download.pytorch.org/whl/cpu --python .venv/bin/python
uv pip install -e . --python .venv/bin/python
.venv/bin/python scripts/server.py
```

Optional extras: append to `-e .` as `-e ".[rapidocr]"` / `-e ".[anki]"`, or for
GPU run `RAG_DEVICE=gpu ./run.sh`.

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

All settings live in `config.json`, read through a central `_paths` module (with
`RAG_ROOT` override for symlinked deployments — see **AGENTS.md**). Most knobs are
tunable via the Setup tab; advanced/internal keys can be edited in config.json directly.

### Core settings

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

### Chat behavior

| Key | Default | Description |
|-----|---------|-------------|
| `chat_mode` | `agentic` | `"agentic"` = tool-calling loop; `"single"` = one-pass retrieval |
| `show_thinking` | `false` | Include model reasoning content in output |
| `context_word_budget` | `3000` | Total words of retrieved context sent to LLM |
| `chunk_word_cap` | `300` | Per-chunk word truncation for flat/child chunks |
| `history_char_budget` | `10000` | Max characters of chat history in context |
| `history_msg_limit` | `12` | Number of past messages included in context |

### Catalog & summarization

| Key | Default | Description |
|-----|---------|-------------|
| `catalog_enabled` | `true` | Enable catalog/list mode |
| `catalog_clarify_threshold` | `20` | Titles threshold before showing "all" prompt |
| `catalog_max_rows` | `200` | Max rows in catalog table |
| `catalog_semantic_pool` | `500` | Pool size for semantic title search |
| `summary_strategy` | `map_reduce` | `"map_reduce"` or `"one_pass"` for book summaries |
| `summary_max_chunks` | `40` | Max chunks for summary material |

### Quiz generation (tuning)

For the full quiz pipeline — planning, audited generation (4-stage parse/grounding/verify/dedupe),
grading analytics, spaced repetition, course view, and exports — see **[quiz.md](quiz.md)**.

Key knobs:

| Key | Default | Description |
|-----|---------|-------------|
| `quiz_default_count` | `10` | Default questions when not specified |
| `quiz_material_chunks` | `10` | Max chunks of retrieved material per prompt |
| `quiz_material_words` | `500` | Word cap per material chunk |
| `quiz_batch_parents` | `4` | Parents sent per generation batch |
| `quiz_parent_word_cap` | `500` | Per-parent word limit during generation |
| `quiz_max_batches` | `12` | Max batches per unit before giving up |
| `quiz_verify_pass` | `true` | Run LLM answer-key verification stage |
| `quiz_grounding_ratio` | `0.85` | Min difflib ratio for excerpt grounding |
| `quiz_dedupe_jaccard` | `0.75` | Stemmed-Jaccard similarity threshold for dedupe |
| `llm_structured_output` | `auto` | Probe-enabled JSON generation (`auto`/`off`/`on`) |
| `quiz_output_format` | `markdown` | Default export format (`markdown`, `gift`, `csv`, `json`, `apkg`) |
| `quiz_grade_llm` | `true` | Enable LLM-assisted grading for short-answer |
| `quiz_sample_children` | `50` | Chunks sampled for books without parent-child indexing |
| `quiz_topic_section_pool` | `120` | Section pool capped when cosine-ranking a topic plan |
| `quiz_topic_max_works` | `3` | Max works a topic plan ranks units across |

### Tuning

The defaults work well out of the box, but several settings are dataset- or hardware-dependent.
For guidance on how RAG parameters behave across different corpora and machines, see
[**tuning.md**](tuning.md) and the detailed reference in **[quiz.md](quiz.md)**.

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
server. Available tools:

- **`search_library`** — hybrid search across your book index
- **`summarize_work`** — retrieve excerpts of a specific book for summarization
- **`list_books`** — catalog-style title listing with topic filtering
- **`make_quiz`** — generate practice questions on a topic or work
- **`plan_quiz` / `build_quiz`** — plan and build persistent, audited quizzes
- **`get_quiz` / `grade_answer`** — view and grade generated quizzes
- **`review_queue` / `study_stats`** — FSRS spaced-repetition review & analytics
- **`get_config` / `set_config`** — read/write editable config keys

```bash
./run_mcp.sh              # stdio transport (for local LM Studio)
./run_mcp.sh --http       # HTTP transport (for remote/LAN access, port 8765)
```

Then in LM Studio: Settings → MCP Servers → Add (Local or Remote) and point it
at the script/command. For named books, `make_quiz` paginates Chroma directly to
gather all matching chunks instead of just the top-k semantically similar ones, so
you get full-book coverage even for books without parent-child indexing.

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

| File / Directory | What it is |
|------------------|------------|
| `config.json` | All settings (embed model, LLM endpoint, chunking, etc.) |
| `config.local.json` | Machine-specific overrides (gitignored) |
| `index/` | ChromaDB vector index |
| `manifest.db` | SQLite database tracking file ingest status + BM25 lexical index |
| `ingest.log` | Full ingest log |
| `conversations/` | Saved chat sessions (JSON) |
| `quizzes/` | Generated quizzes, specs, syllabi, FSRS review store (see **quiz.md**) |
| `.venv/` | Python virtual environment |

Runtime data directories (`index/`, `manifest.db`, logs, `conversations/`, `quizzes/`)
are gitignored. To rebuild from scratch (e.g. after changing the embedding model),
delete `index/` and `manifest.db`, then re-run ingest. The Setup tab also has a
"Force re-embed" option per file.

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

- **GPU is selectable and verified working.** The app auto-detects a usable CUDA
  device (see the Embedding Device setting in the Setup tab: `auto`, `cpu`, or
  `gpu`). Run `RAG_DEVICE=gpu ./run.sh` (or `RAG_CUDA_VERSION=cu126` per your
  toolkit) to install the GPU extras; the app then runs embeddings + reranking on
  CUDA automatically. OCR is always CPU-bound.
- **Fiction classification** depends on Calibre tags. Untagged books default to
  non-fiction.
- **Embedding model tradeoffs:** The default (`multilingual-e5-small`) is fast on
  CPU (~10 files/min). Bigger models (`bge-m3`, `Qwen3-Embedding`) give better
  quality but are 4-10x slower.
- **Portable:** Copy the whole folder to another machine and run `./run.sh`. The
  library itself stays wherever it is.
