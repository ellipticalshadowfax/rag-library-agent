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
- `context_word_budget` — see below; the biggest portability concern.

## The main portability caveat: context budget

`context_word_budget` (`3000` words) and `chunk_word_cap` (`300`) live in
`config.json` (fallback constants in `scripts/agent.py` for older configs).
They were raised from 1000/200 so longer answers, tables, and map-reduce
summaries have room — safe when the deployed model has a large context window:

- On a small model with a tight window (e.g. 1.5B-param / 8k tokens), 3000 words
  of context + history + system prompt can still crowd out the answer. Lower it.
- On a large model with 32k+ context, you can raise it further to gather more
  evidence.

Match the budget to the deployed model's context window. The related history
knobs (`history_char_budget` default 10000, `history_msg_limit` default 12) are
also in `config.json`; long tables/quizzes need the higher budgets so follow-ups
like "explain answer 3" aren't truncated away.

## Catalog / list mode tuning

Catalog requests ("list books about X") are answered by `scripts/catalog.py`
without the LLM. Key knobs:

- `catalog_clarify_threshold` (`20`) — when more than this many titles match, the
  assistant shows the first 20 and waits for **"all"** to expand. Raise it to show
  more rows before asking.
- `catalog_max_rows` (`200`) — hard cap on rows rendered for a full-catalog listing.
- `catalog_semantic_pool` (`500`) — the dense retrieval pool used by the ranking
  fuse. Larger pools give better topic ranking on big libraries but cost more CPU
  per query (still typically sub-second). Chroma clamps it to `collection.count()`.

The three ranking legs (title lexical, dense aggregate, BM25 aggregate) are fused
by reciprocal rank; no LLM is involved, so results are deterministic and cheap.

## Summary strategy tuning

Naming a work ("summarize [Book]") runs `summary_strategy`:

- `"map_reduce"` (default) — a deep, book-wide summary: child chunks are mapped to
  section parents spread evenly across the book, summarized in batches (map calls),
  then combined in one final call (reduce, which streams live). Knobs:
  `summary_max_chunks` (`40`, parents sampled), `summary_context_budget` (`8000`,
  word budget shared across the map batches), `max_tokens_summary` (`4096`).
  Latency is higher (2-4 map calls + 1 reduce) but coverage is far better.
- `"single"` — the old single-pass prompt over a capped excerpt pool. Faster, but
  only samples a handful of chunks.

For small/fast models or short books, `"single"` may be a better latency/quality
trade-off. `summarize_work` agent tool and MCP `summarize_work` honor `deep=true`.


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
| Tight LLM context window | Lower `context_word_budget` |
| Large context window (32k+) | Raise `context_word_budget` |
| CPU-only / slow hardware | Disable `rerank_enabled` |
| Different embedding model | Re-derive `relevance_threshold` via eval |

## Model-size tuning matrix

This project is machine- and model-specific: the shipped defaults (`Default`
column) are tuned for a mid-size local model (roughly the author's ~4B GGUF
with a large context window). If you deploy a **small** (~2B, tight window) or
**large** (~12B+, big window) model, use the suggested values below as a
starting point and validate with `scripts/eval.py`.

| Flag | Default (mid ~4B) | Light (~2B) | Large (~12B+) | Why |
|------|:--:|:--:|:--:|-----|
| `context_word_budget` | `3000` | `1000`–`1500` | `5000`–`8000` | Words of retrieved evidence per prompt. A tight-window 2B model crowds out its answer at 3000; a 12B model can absorb far more evidence. |
| `chunk_word_cap` | `300` | `150`–`200` | `350`–`500` | Per-child-chunk truncation. Lower keeps tiny models inside the window; larger lets big models read fuller sections. |
| `max_tokens_default` | `2048` | `1024`–`1536` | `4096` | Answer length for normal QA. Small models drift on very long generations; big models can write longer answers. |
| `max_tokens_quiz` | `4096` | `2048` | `8192` | Room for a full quiz. A 2B model may truncate a 10-question quiz at 4096. |
| `max_tokens_summary` | `4096` | `2048` | `8192` | Deep-summary reduce step length. |
| `history_char_budget` | `10000` | `4000`–`6000` | `14000`–`20000` | Conversation history injected per turn. Tight windows need less history; big models can hold much more. |
| `history_msg_limit` | `12` | `6`–`8` | `20`–`30` | Prior turns kept. |
| `retrieval_top_k` | `10` | `6`–`8` | `10`–`12` | Chunks retrieved (capped to 8 in code). Small models get confused by too many sources; big models benefit from more. |
| `relevance_threshold` | `0.80` | `0.80`–`0.85` | `0.75`–`0.80` | Low-relevance floor. A stronger-than-default model can tolerate a lower floor (more recall); weak models want a higher guard. |
| `max_retrieval_hops` | `2` | `1` | `2`–`3` | LLM-driven multi-hop search rounds. Each hop costs a generation; small models compound errors, so keep at 1. |
| `rerank_enabled` | `true` | `false` | `true` | Cross-encoder re-scoring (adds latency, not quality). Disable on CPU/slow hardware regardless of size. |
| `agentic_max_steps` | `3` | `2` | `4` | Tool-calling loop turns. Small models drift on long tool chains. |
| `summary_strategy` | `"map_reduce"` | `"single"` | `"map_reduce"` | Deep map-reduce vs single-pass. Map-reduce's many calls are slow and error-prone on a 2B model; single-pass keeps it simple. |
| `summary_max_chunks` | `40` | `12`–`16` | `60`–`80` | Sections sampled for a deep summary (only for `map_reduce`). |
| `summary_context_budget` | `8000` | `3000`–`4000` | `12000`–`16000` | Word budget across map batches. |
| `quiz_default_count` | `10` | `6`–`8` | `12`–`15` | Default questions per quiz. Small models produce better short quizzes. |
| `catalog_semantic_pool` | `500` | `500` | `1000` | Dense pool for catalog ranking (no LLM, CPU-bound — rarely worth tuning by model size). |
| `catalog_clarify_threshold` | `20` | `20` | `20` | When to ask for the full list. Model-independent (LLM-free). |
| `catalog_max_rows` | `200` | `200` | `200` | Full-list cap. Model-independent. |

Guidance when choosing a column:

- **Tight-window vs. large-window** matters more than raw parameter count. A
  ~2B model with a 32k window can safely use the `Large` column for the
  context/history budgets; a ~12B model with only 8k RAM is effectively the
  `Light` column for budget purposes. Match budgets to the **context window**,
  not the parameter count.
- **Start conservative.** Lower budgets/`top_k`/`max_tokens` are always safe;
  raising them only helps if the model stays on-task. Raise one knob at a time
  and watch `eval.py` (and eyeball a few answers).
- **Turn off the expensive stuff first on small models:** `rerank_enabled:
  false`, `max_retrieval_hops: 1`, `summary_strategy: "single"` give the
  biggest latency/quality wins for a 2B CPU deployment.
- These are starting points, not rules — your library's topic mix and your
  model's actual context limit win in the end.

## Changing the embedding model

The embedding model is the biggest single lever on **retrieval quality**, and
switching it changes several *other* settings too. The UI's Embedding-model
picker lists the options (including `bge-m3`, `Qwen3-Embedding`, `bge-small-en`,
`MongoDB/mdbr-leaf-mt`). When you change `embed_model`, revisit these:

| Flag | What to check when you swap the embedder | Why |
|------|-----|-----|
| `embed_dim` | **Must match the new model's output dimension** (e.g. 384 for e5-small, 768 for bge-m3, 1024 for Qwen3-Embedding/bge-m3). Chroma vectors are fixed-dim; a mismatch breaks the index. | Vectors of differing length can't live in one collection. |
| `relevance_threshold` | **Re-derive via `eval.py`.** Embedders produce different distance/score scales — the 0.80 floor tuned for e5-small is meaningless for bge-m3. | The low-relevance guard fires on score; a wrong floor causes false "no match" notes (too high) or misses (too low). |
| `retrieval_top_k` / `context_word_budget` | Better embedders retrieve more-relevant chunks with fewer candidates, so you can often **lower** `top_k` / keep the budget while losing less recall. | A stronger embedder concentrates the good matches at the top. |
| `embed_device` | Bigger embedders (bge-m3 ~2.3GB, Qwen3-Embedding ~1.3GB) are **much slower on CPU**. Move them to GPU (`embed_device: gpu`) or accept the latency hit. | Embedding quality scales with model size, but so does cost. |
| Re-index | **Any `embed_model` or `embed_dim` change requires a full re-index** (old vectors use the old model and can't be compared). The Setup tab's "Force re-embed" handles this — plus the "changing the embedding model requires re-indexing" hint when you apply. | Chroma stores the vectors; it can't tell they came from a different model. |
| `fiction_tags` | No change needed — unaffected by the embedder. | Independent of embedding. |

Practical recipe after a swap:

1. Apply the new `embed_model` (and set `embed_dim` to its dimension).
2. Force a full re-index so every chunk is re-embedded with the new model.
3. Run `scripts/eval.py run` against your `evals/golden.jsonl` and re-derive
   `relevance_threshold` so the low-relevance guard stays correct.
4. Optionally re-tune `retrieval_top_k` / `context_word_budget` down if the
   stronger model concentrates relevant hits at the top.
5. Confirm on-real-questions that distance scores look sane before assuming the
   old numbers transfer.

Note: score-based knobs like `relevance_threshold` are **per-embedder**, so a
tuned `config.json` from one embedder is not portable to another — redo the
eval step after each swap.

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
| `context_word_budget` | `config.json` | `3000` | Total words allowed in the prompt. **The main one to tune** — match your LLM's context window. |
| `chunk_word_cap` | `config.json` | `300` | Per-child-chunk truncation in the prompt. |
| `BM25_TOP_N` | `agent.py` | `30` | BM25 hits taken for the lexical leg before fusion. |
| `BM25_BATCH` | `agent.py` | `20000` | Chunks per paginated build of the lexical index. |
| `FUSE_RRF_K` | `agent.py` | `60` | RRF constant; larger flattens scores and weights the lexical leg more. |
| `MAX_CHUNKS_PER_TITLE` | `agent.py` | `3` | Max chunks kept per distinct title during diversification. |
| pool cap `min(top_k, 8)` / `min(top_k*3, 30)` | `agent.py` | — | Retrieval pool sizing before rerank. |
| `HISTORY_MSG_LIMIT` / `HISTORY_CHAR_BUDGET` | `server.py` | `12` / `10000` | Bound on conversation history sent to the LLM. |
| `MAX_CHUNKS_PER_FILE` / `INGEST_BATCH_SIZE` | `ingest.py` | `5000` / `500` | Oversized-file handling and upsert batching. |

## show_thinking (Qwen3-style thinking models)

Many recent local models (Qwen3 family) emit a `reasoning_content` field during
streaming. By default the web UI **hides** it (the final chat answer streams only
the content). Set `show_thinking: true` to forward the chain-of-thought as part
of the streamed text. This is purely a display choice — the non-streaming path
always falls back to `reasoning_content` if `content` is empty, so answers never
get lost. The Setup tab "Show model thinking" control maps to this key.

## Verify after tuning

After any change, run the compile checks (per `AGENTS.md`):
- `python -m py_compile scripts/*.py`
- `node --check` on the extracted inline JS from `web/index.html`
- `scripts/eval.py diff` to guard against retrieval regressions.
