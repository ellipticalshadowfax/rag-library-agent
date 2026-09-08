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
- `scripts/agent.py` — chat helpers: retrieval, context building, the `SYSTEM_PROMPT`.
- `scripts/ingest.py` — CLI indexer: walks a directory, extracts text, chunks,
  embeds (E5 prefixes), upserts into ChromaDB. Writes `manifest.db`.
- `scripts/scan.py` — CLI dry-run scanner (stats, OCR-need detection).
- `web/index.html` — single-file SPA frontend (Setup/Scan/Ingest/Chat tabs,
  folder-picker dialog, global ingest-activity pill).
- `config.json` — app config (embed model, LLM URL/model, chunking, sets).
- `run.sh` — one-command launcher (venv + deps + model predownload + server).
- `requirements.txt`, `README.md`.

## How it runs
- `./run.sh` (or `RAG_PORT=5000 RAG_HOST=0.0.0.0 ./run.sh`) creates `.venv`,
  installs deps, pre-downloads the embedder, and starts `server.py`.
- Server default: `http://127.0.0.1:5000`. LLM expected at `llm_base_url`
  (default `http://localhost:1234/v1`).
- CLI ingest: `.venv/bin/python scripts/ingest.py /path/to/library --set library`
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
  skips any single file chunking into more than `MAX_CHUNKS_PER_FILE` (default 5000,
  override `INGEST_MAX_CHUNKS`) with a `SKIP (oversized…)` message + manifest record.
- **Embedder prefixes**: E5 models need `passage:` on index and `query:` on retrieve.
  Ingest encodes with `prompt_name="passage"`; retrieval uses `prompt_name="query"`.
- **Deterministic chunk IDs**: `sha256(f"{rel_path}:{index}")` — re-runs upsert in
  place, so `--force`/re-ingest is safe.
- **Fiction awareness**: documents are tagged `fiction`/`nonfiction` from Calibre
  tags; chat adds a warning banner if all retrieved sources are fiction.

## Conventions / requirements
- **NO machine-specific paths or PII.** Use placeholders (`/path/to/your/library`),
  env vars (`RAG_PORT`, `RAG_HOST`, `LMSTUDIO_PORT`), or repo-relative paths
  (`Path(__file__).resolve().parent.parent / "index"`). The collection/set name
  `library` is used as a default identifier throughout — that is fine to keep.
- Default LLM/embed settings live in `config.json`; don't hardcode URLs in scripts.
- Runtime artifacts (`index/`, `manifest.db`, logs, `.venv/`) are gitignored — never
  commit them.
- After editing Python, verify with `python -m py_compile`. After editing
  `web/index.html`'s inline JS, verify with `node --check`.

## When you change shared files
This repo is the canonical source. If you fix something here, the live deployment
copy should be updated to match (see that copy's own `AGENTS.md`), or vice versa,
and the two kept in sync.