# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is
A self-hosted RAG (retrieval-augmented generation) web app that indexes a local
ebook/document library with embedding vectors and lets users ask questions via a
browser UI. Backend is Python/Flask + ChromaDB + Sentence-Transformers; the LLM is
served by a separate local OpenAI-compatible server (LM Studio / llama.cpp).

This repo is the **shareable** copy. The live deployment on the author's machine
has machine-specific paths, indexes, and data that are deliberately **not** in the
repo. Do not introduce them back in.

## Layout
- `scripts/server.py` — Flask backend (web server + all `/api/*` endpoints).
  The web UI's chat streams tokens from `POST /api/chat/stream` (fetch-based
  SSE: `delta`/`done`/`error` events, single-shot, persists via `_persist_chat`).
- `scripts/agent.py` — chat helpers: retrieval, context building, the `SYSTEM_PROMPT`.
- `scripts/agent_loop.py` — bounded agentic tool-calling loop for `/api/chat`:
  lets the LLM call `search_library`, `get_section`, or `summarize_work` tools
  mid-conversation. Falls back to a ReAct text protocol for models without
  function-calling. Non-streaming; the SSE path stays on the single-shot path.
- `scripts/ingest.py` — CLI indexer: walks a directory, extracts text, chunks,
  embeds (E5 prefixes), upserts into ChromaDB. Writes `manifest.db`. OCR-enabled
  ingests can merge OCR text back into the source PDF (`ocr_merge`).
- `scripts/ocr.py` — CLI OCR tool (separate from ingest): OCRs scanned PDFs and
  either merges the text layer back into the original (`--mode merge`) or writes
  sidecar `.txt` files (`--mode sidecar`). Writes `.ocr.lock`, logs to `ocr.log`.
  Driven by the web UI's OCR tab via `/api/ocr*`. `server.py` reaps the OCR
  subprocess via `proc.poll()` (a finished-but-unreaped zombie would otherwise
  make `os.kill(pid,0)` succeed and keep the UI "running" forever) and treats a
  `.ocr.lock` whose recorded PID is dead as stale: cleaned so a new job can start.
- `scripts/ocr_compare.py` — compares Tesseract vs RapidOCR on a page sample of
  every scanned PDF, picks the better engine per file, then OCRs the full file.
  Writes `ocr_compare_report.json`. The OCR engine is chosen by the `ocr_backend`
  config (`tesseract` default, `rapidocr` optional). Tesseract binary discovery:
  `$TESSERACT_BIN` > `shutil.which("tesseract")` (PATH) > `DEFAULT_TESS_BIN` in
  code (a machine-specific fallback — override with the env var on new machines).
  Ingest's OCR path calls `OCR._setup_tesseract()`, so the same resolution
  applies to OCR run from the web UI and the CLI.
- `scripts/merge_ocr_into_pdf.py` — post-OCR utility: adds an invisible
  (searchable) text layer to scanned PDFs from the cached `ocr/<stem>.txt` files.
  Writes in place; only runs when the OCR cache exists and page count matches.
- `scripts/scan.py` — CLI dry-run scanner (stats, OCR-need detection).
- `scripts/chat_store.py` — JSON-file conversation persistence under
  `conversations/` (gitignored). The web UI and the OpenAI-compatible endpoint
  use it to store/manage multiple chats.
- `scripts/mcp_server.py` — MCP server exposing the library index as callable
  tools (`search_library`, `summarize_work`, `list_collections`) so chat clients
  like LM Studio can ground answers in the library. Stdio by default; `--http`
  for a remote/SSE server. Launched via `run_mcp.sh`. Must never write to stdout
  (the stdio JSON-RPC channel) — stray prints are diverted to stderr by
  `_muted_stdout()`; missing collections raise instead of `sys.exit`. Embedder +
  Chroma are cached per-process (loads on CPU).
- `scripts/eval.py` — offline retrieval eval harness (no LLM calls). Scores
  `hit@K`, `mrr@K`, `context_recall`, `context_precision` against
  `evals/golden.jsonl`. Subcommands: `run`, `baseline`, `diff` (CI regression
  gate — exits nonzero if metrics drop beyond tolerance).
- `scripts/bench_embeddings.py` — benchmarks embedding models (speed + quality).
- `scripts/bench_fast.py` — benchmarks fast CPU embedding models on real chunks.
- `scripts/start_lmstudio.sh` — launches LM Studio headless at
  `http://localhost:1234/v1`.
- `web/index.html` — single-file SPA frontend (Setup/Scan/Ingest/Chat tabs,
  folder-picker dialog, global ingest-activity pill).
- `config.json` — app config (embed model, LLM URL/model, chunking, sets).
- `run.sh` — one-command launcher (venv + deps + model predownload + server).
- `run_mcp.sh` — launcher for the MCP server (stdio by default, `--http` for remote).
- `requirements.txt`, `requirements-gpu.txt`, `README.md`.
- `evals/` — `golden.jsonl` (ground truth), `baseline.json`, `results.json`.

## How it runs
- `./run.sh` (or `RAG_PORT=5000 RAG_HOST=0.0.0.0 ./run.sh`) creates `.venv`,
  installs deps, pre-downloads the embedder, and starts `server.py`.
- Server default: `http://127.0.0.1:5000`. LLM expected at `llm_base_url`
  (default `http://localhost:1234/v1`).
- CLI ingest: `.venv/bin/python scripts/ingest.py /path/to/library --set veracrypt1`
  (incremental unless `--force`).

## Key invariants (read before changing)
- **Ingest lock / blocking**: `ingest.py` writes `.ingest.lock` (JSON with PID) on
  start and removes it in a `finally`. `server.py` detects a live ingest from that
  lock and **blocks `/api/chat` (HTTP 409)** while indexing, to avoid ChromaDB
  concurrent read/write errors. Pause/stop is `POST /api/ingest/control` with
  `action: pause|resume|stop` (SIGSTOP / SIGCONT / SIGINT).
- **Status correctness**: `INGEST_STATE` is keyed to the live PID from the lock each
  poll. Never let `finished`/`completed`/`pid` go stale across runs. Progress is
  parsed from the last `[x/y] done=…` line in `ingest.log`.
- **Oversized files**: Chroma upserts have a hard batch limit (~5461). `ingest.py`
  always embeds/upserts in batches (`ingest_batch_size`, default 500) so any single
  file — including huge omnibuses — is processed across multiple calls instead of
  skipped. Files chunking into more than `MAX_CHUNKS_PER_FILE` (default 5000,
  override `INGEST_MAX_CHUNKS`) are flagged as `BIG` and batched; a cheap early
  estimate (chars ÷ tokens) warns before the chunk/embed pass. The collection/set
  name `veracrypt1` is used as a default identifier throughout — that is fine to keep.
- **Embedder prefixes**: E5 models need `passage:` on index and `query:` on retrieve.
  Ingest encodes with `prompt_name="passage"`; retrieval uses `prompt_name="query"`.
- **Deterministic chunk IDs**: `sha256(f"{rel_path}:{index}")` — re-runs upsert in
  place, so `--force`/re-ingest is safe.
- **Fiction awareness**: documents are tagged `fiction`/`nonfiction` from Calibre
  tags; chat adds a warning banner if all retrieved sources are fiction.
- **Hybrid retrieval + rerank**: `server.py` fuses the dense pool (embedding,
  incl. LLM multi-hop) with a BM25 lexical leg via Reciprocal Rank Fusion
  (k=60), then re-scores the fused candidates with a lazy-loaded cross-encoder
  reranker (`rerank_model`, default `cross-encoder/ms-marco-MiniLM-L-6-v2`;
  disable via `rerank_enabled: false`). The BM25 index is built lazily per set
  by paginating `collection.get(limit=20000, offset=…)` (~186k chunks) and
  cached in-process with a `df` token->doc-frequency map.
- **Two tokenizers in `agent.py`**: `stem_tokens()` (English snowballstemmer)
  is used ONLY for the low-relevance guard's term-presence check (small text,
  fast). `tokenize()` (raw lowercase, no stemming) feeds the BM25 index build —
  stemming 185k+ chunks takes ~12 min, so the index must never use stems.
- **Low-relevance guard**: runs on the full candidate pool BEFORE diversification
  (so trimming the context can't drop a decisive term). A missing distinctive
  (non-generic) query term is decisive unless it's a common corpus word
  (`df/corpus ≥ 0.01`) — a single common word like "collect" (dressage jargon vs
  horse books) is tolerated as a synonym-paraphrase, while a rare term like
  "spherification" or a missing proper-noun phrase always flags low relevance.
- **Source display**: the `sources` list dedupes by `(title, source)`, so a
  single multi-chunk book collapses to ONE source entry (e.g. "summarize the
  MindStar book" shows 1 source) even though the context holds up to
  `max_per_title=8` chunks of it — that's intended, not a retrieval failure.
- **Agentic loop**: `agent_loop.py` gives the LLM tool-calling access to
  `search_library`, `get_section`, and `summarize_work`. Controlled by
  `agentic_enabled` and `agentic_max_steps` (default 3) in `config.json`.
  Falls back to a ReAct text protocol (`call: search_library(...)`) for models
  or servers that lack function-calling. Non-streaming only; the SSE streaming
  path in `server.py` stays on the single-shot retrieval path.

## Conventions / requirements
- **NO machine-specific paths or PII.** Use placeholders (`/path/to/your/library`),
  env vars (`RAG_PORT`, `RAG_HOST`, `LMSTUDIO_PORT`, `TESSERACT_BIN`),
  or repo-relative paths
  (`Path(__file__).resolve().parent.parent / "index"`). The collection/set name
  `veracrypt1` is used as a default identifier throughout — that is fine to keep.
- Default LLM/embed settings live in `config.json`; don't hardcode URLs in scripts.
- Runtime artifacts (`index/`, `manifest.db`, logs, `.venv/`, `ocr/`,
  `ocr_compare_report.json`) are gitignored — never commit them.
- After editing Python, verify with `python -m py_compile`. After editing
  `web/index.html`'s inline JS, verify with `node --check`.

## When you change shared files
This repo is the canonical source. If you fix something here, the live deployment
copy should be updated to match (see that copy's own `AGENTS.md`), or vice versa,
and the two kept in sync.
