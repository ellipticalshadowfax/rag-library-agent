# AGENTS.md

Guidance for AI coding agents working in this repository.

## CRITICAL: Symlinked Deployment — always set RAG_ROOT

**This deployment at `/media/tb-desktop/Data/RAG` is symlinked from the source
repo at `/home/tb-desktop/codex/github/rag-library-agent`.** Every top-level
directory (`scripts/`, `web/`, etc.) is a symlink pointing back to the source
code. The **data** (`manifest.db`, `index/`, `config.json`, `quizzes/`,
`conversations/`, `.venv/`) lives under `/media/tb-desktop/Data/RAG`.

`_paths.py` uses `Path(__file__).resolve().parent.parent` to find the "root".
Because `resolve()` follows symlinks, it **always** resolves to the source
repo (`/home/tb-desktop/codex/github/rag-library-agent`) which has no data files
— every script fails with `FileNotFoundError` or imports nothing useful.

**FIX: Always export `RAG_ROOT` before running ANY Python script from the
deployment directory:**

```bash
export RAG_ROOT="/media/tb-desktop/Data/RAG"
```

Every launcher already does this (`run.sh`, `run_mcp.sh`). But ad-hoc scripts,
test scripts, and one-offs do **NOT** — that's why they fail. Always include it:

```bash
export RAG_ROOT="/media/tb-desktop/Data/RAG"
cd "$RAG_ROOT"
.venv/bin/python -c "..."   # or .venv/bin/python scripts/foo.py
```

When `RAG_ROOT` is set, `_paths.rag_root()` returns it, bypassing the symlink
resolution entirely.

### Quick checklist before running anything

1. `export RAG_ROOT="/media/tb-desktop/Data/RAG"` — required, non-negotiable
2. `cd "$RAG_ROOT"` — work from the data directory, not the source repo
3. Use `.venv/bin/python` inside the deployment (it has all deps installed)
4. All scripts use `from _paths import rag_root` — respects the env var

### Current service endpoints

- Web UI / API: `http://127.0.0.1:5000` (Flask `server.py`)
- LLM (llama.cpp): `http://localhost:1234/v1` (model: `qwen2.5-1.5b-instruct`)
- MCP HTTP mode: `--http --port 8765` (see `run_mcp.sh`)
- Collection sets: `veracrypt1` (170k chunks), `nlp_books` (23k chunks)

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
  lets the LLM call `search_library`, `get_section`, `summarize_work`, `list_books`,
  or `make_quiz` tools mid-conversation. Falls back to a ReAct text protocol for
  models without function-calling. Accepts optional `stream_cb`/`progress_cb`: when
  `stream_cb` is provided the final answer is regenerated as ONE streamed chat
  completion (deltas → `stream_cb`, `reasoning_content` filtered unless
  `show_thinking: true`) and a `__done__` marker fires at the end; `progress_cb` is
  called ~every 1.5s during tool resolution to keep SSE alive. With neither
  callback it stays blocking and non-streamed (backward-compatible).
- `scripts/catalog.py` — deterministic catalog/list mode: enumerates
  `manifest.db` titles (`list_books`, cached & invalidated on count change),
  ranks by topic via a 3-leg reciprocal-rank fuse (title lexical, dense
  aggregate, BM25 aggregate) in `find_books`, renders a GFM Markdown table
  (`render_table`, no LLM), and provides list/quiz intent + topic/filter
  extraction. Also hosts `map_reduce_summary` (deep book-wide summaries) and
  `list_sections` (ordered sections of one work). Quiz intent takes priority
  over list intent.
- `scripts/study.py` — structured quiz generation. Produces `Question[]`
  (q, choices, answer, difficulty, type, provenance with title/section/excerpt),
  parses the model's markdown-with-delimiters output back to structure
  (`parse_questions`), and renders via `render_markdown` with collapsible
  `<details>` answers. `generate_quiz` validates every question's
  `provenance.title` against the source titles actually present in the material
  (`source_titles`/`title_matches`), dropping hallucinated citations that name a
  book not in the material. Keep generation/validation and rendering separate so
  future export formats (GIFT/PDF/CSV) only add renderers.
- `scripts/quiz_store.py` — JSON persistence for quiz planning (SESSION 2) and
  generation (SESSION 3) under `quizzes/` (gitignored, mirrors chat_store.py):
  atomic writes, 12-hex ids, helpers to create/get/list/delete specs and
  quizzes, the cached-syllabus cache, and the out-of-scope ledger
  (`quizzes/out_of_scope.json`). Also hosts `inventory_sections` (deterministic
  ordered walk of manifest.db `parents` returning section word counts) and
  `parents_signature` (hash of ordinal/parent_id/words/title so re-ingest
  invalidates a cached syllabus). Also hosts the shared FSRS review store
  (`quizzes/review.json`, keyed by `<quiz_id>:<qid>`) — enroll / due-queue /
  rating helpers — and `weak_unit_signal` (scans finished attempts' analytics
  for repeatedly-missed section titles, feeding the planner's
  weakness-weighted defaults).
- `scripts/quiz_syllabus.py` — `inventory()` (read-only section walk) plus one
  cached LLM outline pass (`build_syllabus`, temperature 0) that turns the
  inventory into units [{unit_id, title, ordinal, parent_ids, subtopics, words}].
  On LLM failure it degrades to a deterministic one-unit-per-section outline so
  the planner never blocks. `get_syllabus()` reads the cache, building on demand.
- `scripts/quiz_plan.py` — the SESSION 2 planner. `plan_quiz(request, set_name,
  cfg, embedder, collection)` resolves scope (work / sections / topic) into an
  editable QuizSpec with a depth knob (`surface|balanced|deep`), allocates
  question counts proportional to section word counts via `allocate_budget`
  (per-unit floor), emits ambiguity warnings, and records out-of-scope refusals
  to the ledger. Topic scope ranks works with `CAT.find_books` and units by
  E5-cosine similarity (no LLM). `revise_spec(spec, deltas)` applies
  conversational edits (include/exclude units, count/difficulty/depth,
  per-unit `counts`) and re-runs allocation. SESSION 3 consumes the spec for
  generation.
- `scripts/quiz_audit.py` — the 4-stage accuracy audit run on every generated
  batch: (0) parse sanity (mcq answer letter present, >=2 unique choices, no
  dup letters; true_false answer in {true,false}; fill_blank/short non-empty),
  (1) grounding (no LLM — `grounding_check` fuzzy-contains the excerpt in the
  cited parent's ACTUAL text via difflib >= `quiz_grounding_ratio`, and checks
  the cited title/section/parent resolve), (2) inline LLM answer-key
  verification (`verify_questions`, only given the grounding parent, batched
  per parent, gated by `quiz_verify_pass`), (3) cross-batch dedupe via
  stemmed-token Jaccard (`is_duplicate`, matches agent.stem_tokens but without
  importing agent). Pure functions; import-light (optional snowballstemmer).
- `scripts/quiz_build.py` — the SESSION 3 generation pipeline. `build_quiz(spec,
  set_name, cfg, client, stream_cb, progress_cb)` resolves units from the
  cached syllabus, walks each unit's `parents` in order (batches of
  `quiz_batch_parents`, trimmed to `quiz_parent_word_cap`) — NEVER semantic
  search — generates via study.py's prompt/parser with `response_format`
  json_schema when the structured-output probe (`llm_structured_output`:
  auto|on|off, cached) says the endpoint supports it, runs the audit inline with
  drop + top-up until the per-unit allocation is met or `quiz_max_batches` is
  exhausted, and appends accepted questions to the quiz file incrementally
  (crash-safe/resumable, `qid` + unit/section tags). Writes a per-unit
  `build_report` with stage drop counts and warnings. Children-sampling
  fallback only when `chunking_strategy != parent_child`.
- `scripts/quiz_grade.py` — the SESSION 4 grading + analytics layer for the
  Study-tab player. `grade_answer(question, response, cfg, client)` is
  deterministic for mcq (letter match), true_false (true/false), and fill_blank
  (normalized token match reusing `study.norm_tokens`), and LLM-assisted for
  short_answer only (structured verdict {score 0..1, feedback}, gated by
  `quiz_grade_llm`, using just the question's provenance excerpt, with a
  token-overlap fallback on failure). `compute_analytics(attempt, questions,
  weak_threshold)` aggregates accuracy per unit/difficulty/type (weak unit =
  accuracy < `quiz_weak_threshold`) and builds provenance-linked study points
  for missed questions. Pure/import-light, mirroring quiz_audit.
- `scripts/quiz_export.py` — SESSION 6 export renderers over a stored quiz's
  Question[]: GIFT (Moodle), CSV, JSON, and Anki `.apkg` via genanki (optional
  `try/except` import — document the optional dependency). Pure functions
  (`export_quiz` returns payload+mime+is_binary); wired to
  `GET /api/quiz/<id>/export?format=...` and to the `quiz_output_format`
  config key as its default.
- `web/vendor/` — vendored `marked.min.js` (v12) + `purify.min.js` (DOMPurify
  v3) with `LICENSES.md`. Assistant turns render Markdown through
  `renderMarkdown()` (marked → DOMPurify sanitize, allowlist includes
  `<details>/<summary>`); user turns stay escaped. Committed (no CDN needed).
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
  tools (`search_library`, `summarize_work`, `list_collections`, `list_books`,
  `make_quiz`) so chat clients like LM Studio can ground answers in the library.
  Stdio by default; `--http`
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
  folder-picker dialog, global ingest-activity pill). Assistant turns render
  Markdown via the vendored marked + DOMPurify (`renderMarkdown`); the Setup
  tab edits all chat/catalog/summary/quiz tuning keys.
- `config.json` — app config (embed model, LLM URL/model, chunking, sets, and
  the chat/catalog/summary/quiz knobs: `chat_mode`, `show_thinking`,
  `context_word_budget`, `chunk_word_cap`, `history_*`, `catalog_*`,
  `summary_*`, `quiz_*`, `max_tokens_*`). Missing keys default in
  `server.default_cfg()`.
- **Config registry**: `server.CONFIG_META` is the single source of config
  *metadata* ({label, group, min, max, advanced, description, editable} for
  every key); `server.default_cfg()` is the single source of *defaults*. The
  `POST /api/config` allowlist is derived from `CONFIG_META` (writable =
  `editable` keys only) so it can never drift; `GET /api/config/schema`
  returns the full registry. Advanced/internal retrieval keys (`fuse_rrf_k`,
  `bm25_top_n`, `bm25_batch`, `bm25_min_coverage`, `rerank_top_n`,
  `max_chunks_per_title`, `dense_pool_multiplier`, `fused_pool_floor`,
  `lexical_backend`, `quiz_material_chunks`/`quiz_material_words`) are
  tool-tweakable in config.json but NOT rendered in the Setup tab; their
  defaults match the old hardcoded constants so retrieval is bit-identical.
  Use `server.cfg_int`/`cfg_float` to read numeric keys with clamping.
- **Known dead knobs**: none remain — SESSION 6 wired both. `quiz_output_format`
  now sets the default for `GET /api/quiz/<id>/export?format=...`, and the
  agentic `make_quiz` `kind` param routes large/stored quiz requests through the
  planner + audited builder (see the "Quiz course & exports (SESSION 6)"
  invariant below).
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
  cached in-process with a `df` token->doc-frequency map. The index is hydrated
  from persistent `bm25_tokens`/`bm25_df` tables in `manifest.db` (written at
  ingest); if the persisted index covers <90% of the collection's chunks
  (`BM25_MIN_COVERAGE`, checked via `_bm25_is_covered`) it's treated as stale and
  rebuilt from Chroma automatically. Repair a one-off incomplete posting list
  with `ingest.py --rebuild-bm25 --set <set>` (uses `Manifest.replace_bm25`
  bulk writer).
- **Two tokenizers in `agent.py`**: `stem_tokens()` (English snowballstemmer)
  is used ONLY for the low-relevance guard's term-presence check (small text,
  fast). `tokenize()` (raw lowercase, no stemming) feeds the BM25 index build —
  stemming 185k+ chunks takes ~12 min, so the index must never use stems.
- **Low-relevance guard**: runs on the full candidate pool BEFORE diversification
  (so trimming the context can't drop a decisive term). Only a **proper-noun**
  signal is decisive: a missing capitalized phrase ("Bilbo Baggins", "Captain
  Ahab") or a missing proper-noun single word (e.g. "Gulliver", "Cthulhu") that
  is NOT a common corpus word (`df/corpus ≥ 0.01`). Ordinary descriptive words
  ("dystopian", "opulent", "cultist") are deliberately only a soft signal and
  never fire alone — authors paraphrase such vocabulary even when retrieval is
  correct, and treating them as decisive produced false "no match" notes that
  drove the model to re-search endlessly and drift off track.
- **Source display**: the `sources` list dedupes by `(title, source)`, so a
  single multi-chunk book collapses to ONE source entry (e.g. "summarize the
  MindStar book" shows 1 source) even though the context holds up to
  `max_per_title=8` chunks of it — that's intended, not a retrieval failure.
- **Agentic loop**: `agent_loop.py` gives the LLM tool-calling access to
  `search_library`, `get_section`, `summarize_work`, `list_books`, and
  `make_quiz`. Controlled by `agentic_enabled` and `agentic_max_steps`
  (default 3) in `config.json`. Falls back to a ReAct text protocol
  (`call: search_library(...)`) for models or servers that lack
  function-calling. When `stream_cb` is provided the final answer is
  regenerated as ONE streamed completion (`reasoning_content` filtered unless
  `show_thinking: true`) and the SSE path uses it with `progress_cb`
  heartbeats during tool steps; otherwise it stays blocking and non-streamed.
- **Catalog/list + quiz modes**: queries like "list books about X" and "quiz
  me on Y" are detected deterministically (in `catalog.py`) in `server.py`
  BEFORE retrieval/agent loop, and answered without passage retrieval. List
  results render via `CAT.render_table` (no LLM); quizzes via `study.py`
  (structured `Question[]` + collapsible `<details>` answers). Quiz intent
  takes priority over list. `chat_mode` ("agentic" default vs "single")
  controls whether the web SSE path runs the streaming agentic loop or the
  old single-shot path. The catalog cache in `catalog.list_books` is
  invalidated when `collection.count()` changes. Map-reduce deep summaries
  (`summary_strategy: "map_reduce"`) run only for named works (title_mode).
- **Quiz planning (SESSION 2)**: the Study tab's builder plans an editable
  QuizSpec through `POST /api/quiz/plan` (scope: work / sections / topic),
  `POST /api/quiz/plan/revise` (deltas: include/exclude units, count,
  difficulty, depth, per-unit `counts`), and `POST /api/quiz/syllabus`.
  Work/sections scope resolves units from the deterministic `parents` walk; only
  topic scope needs the E5 embedder (cosine) — no LLM for planning. Budget
  allocation is proportional to section word counts with a per-unit floor.
  Persisted specs live under `quizzes/specs/`, cached syllabi under
  `quizzes/syllabus/<set>/<slug>.<parents_hash>.json` (invalidated on re-ingest),
  and refusals under `quizzes/out_of_scope.json`.
- **Quiz generation (SESSION 3)**: `POST /api/quiz/generate` (SSE, worker-thread
  + queue bridge, blocked by the ingest lock like `/api/chat`) builds a quiz from
  a spec. Material is an ordered, batched walk of the `parents` table — never
  semantic search. Each generated question must clear the 4-stage audit
  (`scripts/quiz_audit.py`: parse sanity, parent grounding, inline LLM
  answer-key verify, cross-batch dedupe) before it is kept; the builder drops
  and tops up until each unit's allocation is met or `quiz_max_batches` is
  exhausted. Questions are appended incrementally (crash-safe/resumable) with a
  `qid` + unit/section tags, and the full `build_report` (per-unit drop counts,
  per-stage totals, structured-output flag, warnings) is stored on the quiz.
  Quiz CRUD: `GET /api/quizzes`, `GET /api/quiz/<id>`, `DELETE /api/quiz/<id>`.
  Structured output is gated by the `llm_structured_output` probe (auto|on|off,
  cached in-process, falls back to the delimiter protocol).
- **Quiz player & grading (SESSION 4)**: the Study-tab player takes a stored quiz
  one question at a time. `POST /api/quiz/<id>/attempt` returns the questions
  with `answer` stripped plus an `attempt_id`; `POST /api/quiz/<id>/answer`
  grades one pre-selected question via `grade_answer` (mcq letter match,
  true_false bool, fill_blank token match; short_answer LLM-assisted when
  `quiz_grade_llm`, else token overlap) and stores the grade on the attempt;
  `POST /api/quiz/<id>/finish` runs `compute_analytics` (per-unit/difficulty/
  type accuracy, weak units < `quiz_weak_threshold`, provenance-linked study
  points) and marks the attempt finished; `GET /api/quiz/<id>/analytics`
  returns a finished attempt's analytics. Attempts (answers, scores, ts,
  analytics) are persisted INSIDE the quiz file under `quiz["attempts"]` so an
  interrupted session survives a restart. Grading never uses the whole context —
  only the question's own provenance excerpt.
- **Quiz review & FSRS (SESSION 5)**: missed questions from a finished attempt
  are enrolled into a single shared `quizzes/review.json` keyed by
  `<quiz_id>:<qid>` (FSRS card snapshot + full question incl. answer + a review
  `history`). The Study tab's Review pane pulls `GET /api/review/queue?set=`
  (due cards across quizzes, oldest due first), applies a rating via
  `POST /api/review/<quiz_id>/<qid>` (`again|hard|good|easy`), and shows counts,
  retention, and a strength-over-time series via `GET /api/review/stats`;
  `GET /api/review/summary` returns per-quiz due counts for the quiz-list badge.
  Scheduling is `py-fsrs` (`fsrs==6.3.2` in requirements.txt) with UTC now and
  is gated by `fsrs_enabled`; the review queue is capped by `quiz_review_limit`.
  The planner applies weakness-weighted builder defaults: units whose section
  titles recur in `analytics.study_points` get a weight > 1 in
  `allocate_budget`, surfaced as a spec warning the user can override.
- **Quiz course & exports (SESSION 6)**: the Study tab's Course pane browses a
  work's syllabus via `GET /api/course?set=&work=` (units, word counts, and a
  per-unit `quizzed` flag from stored quizzes), a whole-work reading summary via
  `POST /api/course/summary` (reuses `catalog.map_reduce_summary`), per-unit
  summaries via `POST /api/course/unit-summary` (LLM over that unit's parents),
  and `POST /api/course/quiz-unit` (plans a sections-scope spec for one unit).
  Stored quizzes export via `GET /api/quiz/<id>/export?format=...` — markdown,
  gift, csv, json, apkg (genanki optional) — with `quiz_output_format` as the
  default format when no `format` param is given (`scripts/quiz_export.py`).
- **Quiz conversational negotiation (SESSION 6)**: larger quiz requests (or
  those phrasing "plan/build/save") route through `quiz_plan.plan_quiz` in
  `maybe_quiz` to produce an editable spec proposal with numbered options,
  persisted as the conversation's `pending_quiz_spec` (`chat_store`). Follow-up
  messages ("drop unit 1", "double unit 2", "make it 30", "harder", "deep",
  "build it") apply deltas via `quiz_plan.revise_spec`; the small/default inline
  `study.generate_quiz` path is unchanged (no regression).
- **Agentic + MCP quiz surface (SESSION 6)**: `agent_loop.py` tools now include
  `plan_quiz`, `build_quiz`, `get_quiz`, `grade_answer`, `review_queue`, and
  `study_stats`, and `make_quiz` accepts `kind=quick|stored` (large counts or
  `stored` route through the planner + audited builder and return a stored quiz
  id + summary). `mcp_server.py` exposes the same `plan_quiz`/`build_quiz`/
  `get_quiz`/`grade_answer`/`review_queue` plus `get_config`/`set_config`
  (editable keys mirrored from the SESSION 1 registry, written to the gitignored
  `config.local.json`); all runs are wrapped so the stdio channel stays clean.
- **Eval topic->works discovery (SESSION 6)**: `scripts/eval.py` now also scores
  catalog discovery. Golden records with `"mode": "topic"` + `expected_works`
  run `catalog.find_books` (no LLM) and produce `catalog_hit@5`,
  `catalog_hit@10`, `catalog_recall`. `run` writes both `metrics` (retrieval,
  over retrieval records only) and `topic_metrics`; `diff` gates retrieval keys
  and, when the baseline contains them, the topic keys. Adding/removing topic
  records changes the golden set, so refresh `baseline.json` deliberately.

## Conventions / requirements
- **NO machine-specific paths or PII.** Use placeholders (`/path/to/your/library`),
  env vars (`RAG_PORT`, `RAG_HOST`, `LMSTUDIO_PORT`, `TESSERACT_BIN`),
  or repo-relative paths
  (`Path(__file__).resolve().parent.parent / "index"`). The collection/set name
  `veracrypt1` is used as a default identifier throughout — that is fine to keep.
- Default LLM/embed settings live in `config.json`; don't hardcode URLs in scripts.
- Runtime artifacts (`index/`, `manifest.db`, logs, `.venv/`, `ocr/`,
  `conversations/`, `quizzes/`, `ocr_compare_report.json`) are gitignored —
  never commit them.
- After editing Python, verify with `python -m py_compile`. After editing
  `web/index.html`'s inline JS, verify with `node --check`.

## When you change shared files
This repo is the canonical source. If you fix something here, the live deployment
copy should be updated to match (see that copy's own `AGENTS.md`), or vice versa,
and the two kept in sync.
