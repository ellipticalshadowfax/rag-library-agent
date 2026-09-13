# Tuning & portability

This document explains how the RAG tuning parameters behave across different
hardware and datasets. The defaults are sensible out of the box, but several
settings are dataset- or hardware-dependent and worth re-tuning for a specific
library or machine. Use `scripts/eval.py` to measure the impact of any change
instead of guessing.

## How portable are the defaults?

**Robust on any setup (no change needed):**
- `chunk_tokens: 330`, `chunk_overlap: 60` — solid mid-range values. Only
  extremes (very dense technical text, or short-entry reference books) benefit
  from adjustment.
- `relevance_threshold: 0.8` — acts as a *guard* that flags low relevance; it
  never drops results, so it degrades gracefully on weak datasets.
- `rerank_model` / `rerank_enabled` — language-agnostic; only affects latency.
- `max_retrieval_hops: 2`, `retrieval_hops_driver: llm` — self-limiting and safe.

**Hardware-dependent (portable, but will *feel* different):**
- `embed_device: cpu` — correct everywhere; just slower on CPU, faster on GPU.
  No correctness change.
- `rerank_enabled: true` — the main latency cost on a weak CPU. Consider
  disabling on low-end hardware.

**Dataset-dependent (should be re-tuned per corpus):**
- `retrieval_top_k` — capped to 8 in code. Small corpora may want fewer; large
  omnibus-heavy libraries may want more.
- `relevance_threshold` — score distributions vary by embedding model and
  corpus, so the "right" floor differs per setup.
- `CONTEXT_WORD_BUDGET` — see below; the biggest portability concern.

## The main portability caveat: context budget

`CONTEXT_WORD_BUDGET` (`1500` words) and `CHUNK_WORD_CAP` (`240`) are **hard-coded**
in `scripts/agent.py`, not in `config.json`. They assume a decently sized local
context window:

- On a small model with a tight window (e.g. 1.5B-param / 8k tokens), 1500 words
  of context + history + system prompt can crowd out the answer. Lower it.
- On a large model with 32k+ context, you're leaving evidence-gathering capacity
  unused. Raise it.

Match the budget to the deployed model's context window.

## When parent-child chunking helps (and when it silently doesn't)

`chunking_strategy: parent_child` retrieves small child chunks but generates from
larger section-level parent texts. This shines on books with real structure
(chapters, headings). On datasets without detectable structure (scanned PDFs,
unstructured notes) it **silently falls back to flat chunks** — it won't break,
but the benefit disappears.

## Suggested starting points for a new dataset

| Situation | Recommended change |
|-----------|-------------------|
| Short / technical / reference content | `chunk_tokens` 200–250 |
| Long-form fiction | `chunk_tokens` 400+ |
| Tight LLM context window | Lower `CONTEXT_WORD_BUDGET` |
| Large context window (32k+) | Raise `CONTEXT_WORD_BUDGET` |
| CPU-only / slow hardware | Disable `rerank_enabled` |
| Different embedding model | Re-derive `relevance_threshold` via eval |

## Measuring changes

Prefer data over intuition. The offline harness in `scripts/eval.py` scores
`hit@K`, `mrr@K`, `context_recall`, and `context_precision` against your own
`evals/golden.jsonl`. The `diff` subcommand acts as a CI regression gate that
exits nonzero if metrics drop beyond tolerance.

## CLI-passed parameters

Beyond `config.json`, several tools accept tuning knobs on the command line.
These override their config counterparts only for that invocation.

**`scripts/agent.py`** — CLI chat agent.
| Flag | Default | Notes |
|------|---------|-------|
| `--set` | `veracrypt1` | Collection to query |
| `--top-k` | `10` | Retrieval results (capped at 8 in code) |
| `--model` | config `llm_model` | Override the LLM model id |

**`scripts/eval.py`** — retrieval eval harness (offline; no LLM calls).
| Flag | Default | Notes |
|------|---------|-------|
| `run --top-k` | `10` | Retrieval depth scored against your golden set |
| `run --set` | all | Restrict eval to one library set |
| `diff --tolerance` | `0.05` | Max allowed per-metric drop before CI gate fails |

**`scripts/ingest.py`** — indexer (mostly non-tuning, but worth noting).
| Flag | Default | Notes |
|------|---------|-------|
| `--set` | `veracrypt1` | Collection/set name to write into |
| `--force` | off | Reprocess every file (ignore manifest mtime/size) |
| `--only` | — | Only process paths containing this substring |

**`scripts/ocr.py`** — standalone OCR tool.
| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `merge` | `merge` writes text layer into the PDF; `sidecar` writes `.txt` |
| `--backend` | config `ocr_backend` | `tesseract` or `rapidocr` |
| `--languages` | config `ocr_languages` | Comma-separated OCR languages |
| `--force` | off | OCR even if a text layer exists |

**`scripts/ocr_compare.py`** — compares Tesseract vs RapidOCR per page and picks the
better engine, then OCRs each file.
| Flag | Default | Notes |
|------|---------|-------|
| `--sample` | `5` | Pages sampled per file for the comparison |
| `--limit` | `0` | Max files to process (0 = all) |
| `--tess-threshold` | `0.50` | Min Tesseract confidence to trust a page |
| `--rapid-threshold` | `0.45` | Min RapidOCR confidence to trust a page |
| `--workers` | `8` | Parallel page OCR workers (tune to your cores/RAM) |
| `--force` | off | Re-OCR files that already have a cache |
| `--force-tesseract` | off | Force Tesseract for every file |
| `--sample-only` | off | Compare only; do not OCR full files |

**`scripts/mcp_server.py`** — MCP server (stdio by default).
| Flag | Default | Notes |
|------|---------|-------|
| `--http` | off | Serve over HTTP/SSE instead of stdio |
| `--host` | `127.0.0.1` | Bind host for `--http` |
| `--port` | `8765` | Port for `--http` |

## Environment variables

| Variable | Used by | Default | Notes |
|----------|---------|---------|-------|
| `RAG_PORT` | `server.py` / `run.sh` | `5000` | Web app listen port |
| `RAG_HOST` | `server.py` / `run.sh` | `127.0.0.1` | Web app bind host (`0.0.0.0` for LAN) |
| `RAG_DEVICE` / `RAG_GPU` | `run.sh` | `cpu` | `cpu` or `gpu` dependency/install mode |
| `INGEST_BATCH_SIZE` | `ingest.py` | `500` | Embed/upsert batch per call (Chroma hard limit ~5461). Lower for low-RAM machines. |
| `INGEST_MAX_CHUNKS` | `ingest.py` | `5000` | Chunk count that flags a file "big" for batched processing |
| `TESSERACT_BIN` | `ocr_compare.py` / `ocr.py` | PATH (`shutil.which`) | Explicit path to the tesseract binary. Resolved in order: `TESSERACT_BIN` env > PATH (`shutil.which`) > in-code default (machine-specific — set this env var on new machines to override) |
| `OCR_SAMPLE_DIR` | `ocr_compare.py` | `/tmp/opencode/ocr_samples` | Where page samples are cached |

`CUDA_VISIBLE_DEVICES=""` is forced in the Python scripts to pin embeddings to
CPU regardless of install mode.

## Hard-coded in-code parameters (not in config.json)

These live in `scripts/agent.py` and `scripts/ingest.py` and may need editing
for unusual setups:

| Constant | File | Value | What it controls |
|----------|------|-------|------------------|
| `CONTEXT_WORD_BUDGET` | `agent.py` | `1500` | Total words allowed in the prompt. **The main one to tune** — match your LLM's context window. |
| `CHUNK_WORD_CAP` | `agent.py` | `240` | Per-child-chunk truncation in the prompt. |
| `BM25_TOP_N` | `agent.py` | `30` | BM25 hits taken for the lexical leg before fusion. |
| `BM25_BATCH` | `agent.py` | `20000` | Chunks per paginated build of the lexical index. |
| `FUSE_RRF_K` | `agent.py` | `60` | RRF constant; larger flattens scores and weights the lexical leg more. |
| `MAX_CHUNKS_PER_TITLE` | `agent.py` | `3` | Max chunks kept per distinct title during diversification. |
| pool cap `min(top_k, 8)` / `min(top_k*3, 30)` | `agent.py` | — | Retrieval pool sizing before rerank. |
| `HISTORY_MSG_LIMIT` / `HISTORY_CHAR_BUDGET` | `server.py` | `12` / `6000` | Bound on conversation history sent to the LLM. |
| `MAX_CHUNKS_PER_FILE` / `INGEST_BATCH_SIZE` | `ingest.py` | `5000` / `500` | Oversized-file handling and upsert batching. |