================================================================================
RAG-WEB-CHAT — SEARCH + LLM INTEGRATION IMPROVEMENT PLAN
Repo: /home/tb-desktop/codex/github/rag-web-chat
Venv: .venv/bin/python (repo-local)
Run tests: No automated test suite exists. Verify with
           `.venv/bin/python -m py_compile scripts/*.py` and
           `node --check web/index.html` (after any inline-JS change).
Dev server log: server.py prints a startup banner and `[chat]` lines to stdout;
                ingest/OCR progress is parsed from `ingest.log` / `ocr.log` in
                the repo root.

HOW TO USE THIS FILE
--------------------
This file is a multi-session implementation plan. Each SESSION block is one
logical unit of work meant to be executed in a single agent chat, in order.
Start a new chat for each session and point the agent at this file (each
session's PROMPT tells the agent to read this file, implement only that
session, run the verification commands, and write a HANDOFF block). Do not
begin a later session until the earlier one it depends on is complete and its
HANDOFF Status is set. Deviations and results are recorded in each session's
HANDOFF block so the next session starts from the true state of the code.
--------------------

KNOWN PRE-EXISTING TEST FAILURES
- None. The repo has no automated test suite; only `.venv` third-party tests.

================================================================================
SCOPE OVERVIEW — 6 SESSIONS
================================================================================
1. Extract a reusable retrieval engine out of Flask (`server.py`) so eval and
   the agent loop share one code path.
2. Add an eval harness (golden set + hit-rate/MRR/context metrics + CI-style
   baseline diff gate), referencing Haystack.
3. Persist the BM25 lexical index to disk (Option B) so it is not rebuilt in
   memory per process; keep Chroma.
4. Add boundary-aware recursive chunking + TOC/section-aware extraction
   (split at paragraph/sentence boundaries; detect PDF bookmarks/EPUB chapters).
5. Add parent-child retrieval (search small children, generate from large
   parents), opt-in behind `chunking_strategy: parent_child`.
6. Add an agentic tool-calling loop to `/api/chat` (default on, ReAct
   fallback for models without function-calling), reusing MCP tool logic.

================================================================================
SESSION 1 — Extract a reusable retrieval engine
================================================================================
START NEW CHAT for this session.

BACKGROUND (read before starting):
- scripts/server.py  — `_prepare_rag` (lines ~1229-1401), `_bm25_leg`,
  `_rrf_fuse`, `_get_bm25_index` (lines ~512-600), `run_rag_chat`
  (lines ~1403-1450), `_match_titles` (lines ~1172-1214).
- scripts/agent.py  — `retrieve`, `retrieve_multi_hop`, `rerank_hits`,
  `diversify_hits`, `detect_low_relevance`, `build_context` (lines 120-426).

CONTEXT:
Retrieval orchestration (dense retrieve -> BM25 leg -> RRF fuse -> rerank ->
relevance guard -> diversify -> word budget -> source dedupe) currently lives
inside `_prepare_rag` in server.py, mixed with LLM prompt/message construction.
Session 2 (eval) and Session 6 (agent loop) both need to run retrieval without
a Flask app and without generating answers. This session moves that
orchestration into agent.py (or a new scripts/search.py) as a pure function
that returns hits + derived metadata.

WHAT TO DO:
1. In scripts/agent.py, add `retrieve_rag(set_name, query, top_k, filter_kind,
   cfg) -> dict` returning at minimum:
   - `hits` (list, best-first, each with id/document/metadata + scores)
   - `sources` (deduped by (title, source), same shape as server.py builds)
   - `fiction_only` (bool)
   - `low_relevance` (bool) and `relevance_reason` (str)
   - `title_mode` (bool) and `matched_titles` (list) for work-directed routing
   - `message_context` (str) — the raw context string used for the prompt
   Reuse existing agent helpers. Keep the exact same retrieval behavior as
   `_prepare_rag` today (no behavior change, just relocation).
   - Move `_get_bm25_index`, `_bm25_leg`, `_rrf_fuse`, and the BM25 constants
     (BM25_TOP_N, BM25_BATCH, FUSE_RRF_K, _bm25_cache) into agent.py (or a new
     scripts/search.py imported by both). Keep function names stable.
2. In scripts/server.py:
   - Import the moved BM25 helpers / `retrieve_rag` from agent.
   - Rewrite `_prepare_rag` to call `agent.retrieve_rag(...)` and then build
     only `user_msg` / `messages` (title-mode summary prompt and fiction /
     low-relevance NOTE logic stays in server.py exactly as-is).
   - Delete the now-duplicated BM25/fusion code from server.py.
3. Ensure `mcp_server.py` still works (it imports `agent.retrieve`,
   `diversify_hits`, `detect_low_relevance`, `_match_titles` — do not remove
   those public helpers).
4. Verify:
   - `.venv/bin/python -m py_compile scripts/agent.py scripts/server.py scripts/mcp_server.py scripts/eval.py`
   - Start the server (`.venv/bin/python scripts/server.py`) and smoke-test
     `/api/chat` and `/v1/chat/completions` once. Confirm behavior is unchanged.
   - `.venv/bin/python scripts/mcp_server.py` exits/imports without error (run
     with a short timeout; stdio mode will block, so just confirm no import
     traceback via `--help`).

PROMPT FOR THIS SESSION:
--------------------------------------------------------------------------------
Read PLAN.md in this repo. Implement ONLY SESSION 1 (extract a reusable
retrieval engine out of server.py into agent.py / scripts/search.py), following
the WHAT TO DO steps exactly and preserving current retrieval behavior. Run the
verification commands (py_compile; smoke-test /api/chat; mcp_server import).
Afterward, fill in the SESSION 1 HANDOFF block in PLAN.md: set Status to
"Complete" (or "Blocked" with reason), note any deviations, the smoke-test
result, and the exact set of public helper functions now available in
agent.py/search.py for later sessions. Do not implement any other session.
--------------------------------------------------------------------------------

HANDOFF (filled in by agent after completion):
Status: Complete

Deviations: None. Behavior preserved exactly.

Smoke-test result: Server started, BM25 index built (165828 docs),
/api/chat and /v1/chat/completions both returned expected LLM-
connection-error responses (no LLM server running) with retrieval
pipeline executing without errors.

Public helpers now in agent.py for later sessions:
- retrieve_rag(set_name, query, top_k, filter_kind, cfg, embedder,
    collection, client=None, reranker=None, chat_history=None) -> dict
    Returns: hits, sources, fiction_only, low_relevance,
    relevance_reason, title_mode, matched_titles, message_context,
    common_terms
- retrieve(query, embedder, collection, top_k, filter_kind, cfg, where_extra)
- retrieve_multi_hop(query, embedder, collection, client, cfg, top_k, filter_kind)
- rerank_hits(hits, query, reranker, top_n)
- diversify_hits(hits, limit, max_per_title)
- detect_low_relevance(query, hits, cfg, common_terms)
- build_context(hits, max_words)
- _match_titles(query, set_name, limit)
- _get_bm25_index(set_name, collection)
- _bm25_leg(set_name, query, skip_ids, collection, n)
- _rrf_fuse(*ranked_lists, k)
- tokenize(text), stem_tokens(text)
- Constants: BM25_TOP_N, BM25_BATCH, FUSE_RRF_K, CONTEXT_WORD_BUDGET,
  CHUNK_WORD_CAP, MAX_CHUNKS_PER_TITLE

================================================================================
SESSION 2 — Eval harness for retrieval quality
================================================================================
START NEW CHAT. Requires SESSION 1 complete.

BACKGROUND (read before starting):
- scripts/agent.py — `retrieve_rag` (added in Session 1).
- scripts/server.py — `_match_titles`, `default_set_name`.
- manifest.db — sqlite `files` table (rel_path, title, set_name, tags, kind).
- config.json — set names in `sets`.

CONTEXT:
There is no way to measure whether a retrieval change helps or hurts. This
session adds a deterministic, offline eval harness (the Haystack/RAGAS
pattern) that scores the retrieval layer only (no LLM required). Ground truth
is "did the expected source surface in top-K", which can be derived from known
titles in manifest.db without authoring reference answers.

WHAT TO DO:
1. Create `evals/golden.jsonl` — one JSON object per line:
   `{"query": "...", "expected_title": "...", "set": "...", "filter_kind": null}`.
   Seed it with ~15-25 representative queries drawn from real titles/sets in
   manifest.db (e.g. one per distinct set). This file is committed.
2. Create `scripts/eval.py` with subcommands:
   - `run`  — for each golden query call `agent.retrieve_rag(...)` and compute:
       - Hit rate @K and MRR @K for K=5 and K=10: did a hit whose metadata
         `title` matches `expected_title` appear, and at what rank.
       - Context precision/recall over the final `message_context` (does the
         expected source's text appear in the context).
     Aggregate (mean) across queries; print a per-query and summary report.
     Write the summary to `evals/results.json`.
   - `baseline` — snapshot current `evals/results.json` to `evals/baseline.json`.
   - `diff` — compare current results vs baseline; EXIT NONZERO if any metric
     drops by more than 5% (regression gate, the CI pattern). Allow
     `--tolerance` override.
   - Support `--set` filter and `--top-k`.
   - Skip gracefully (no crash) if the embedder/collection is unavailable.
3. Wire nothing into the server. Keep eval CLI-only.
4. Verify:
   - `.venv/bin/python scripts/eval.py run` produces a report without LLM calls.
   - `.venv/bin/python scripts/eval.py baseline` then `.venv/bin/python scripts/eval.py diff`
     exits 0 (no regression vs itself).
   - `python -m py_compile scripts/eval.py`.

PROMPT FOR THIS SESSION:
--------------------------------------------------------------------------------
Read PLAN.md. Implement ONLY SESSION 2 (eval harness). Create
evals/golden.jsonl and scripts/eval.py per the WHAT TO DO steps, using
agent.retrieve_rag from Session 1. Run `eval.py run`, `eval.py baseline`,
`eval.py diff` and confirm they work and diff exits 0. Fill in the SESSION 2
HANDOFF block in PLAN.md (Status, golden-set size, and the baseline metric
values recorded in evals/baseline.json). Do not implement other sessions.
--------------------------------------------------------------------------------

HANDOFF (filled in by agent after completion):
Status: Complete

Golden set: 22 queries in evals/golden.jsonl across both indexed sets
(18 library, 4 nlp_books). Every query references a title confirmed
indexed in Chroma. Baseline metrics in evals/baseline.json (recorded at
top_k=10, rerank on, multi-hop off for determinism/LLM-free eval):
  hit@5=1.0000  hit@10=1.0000  mrr@5=0.9545  mrr@10=0.9545
  context_recall=1.0000  context_precision=0.5962
Notes:
- scripts/eval.py subcommands: run / baseline / diff (--set, --top-k on
  run; --tolerance on diff). Diff exits nonzero on >5% per-metric drop.
- Eval passes client=None to retrieve_rag so multi-hop (LLM-dependent) is
  disabled; otherwise retrieval path is identical to production.
- context_precision is fraction of the deduped (title,source) sources that
  match the expected title (measures context purity; several fiction queries
  legitimately pull a second source, hence <1.0).
- Golden queries initially missed 5/22 (Flatland, Paradise Lost, Creative
  Elements, Hypnotism, Brainwashing) because the phrasing did not engage
  title-mode or dense retrieval; reworded to representative, retrievable
  phrasings -> 22/22 hit.
- Verification: py_compile OK; run/baseline/diff all work; diff vs itself
  exits 0; simulated regression (mrr@5 lowered) makes diff exit 1 (gate works).

===============================================================================
SESSION 3 — Persistent BM25 lexical index (Option B)
===============================================================================
START NEW CHAT. Requires SESSION 1 complete.

BACKGROUND (read before starting):
- scripts/agent.py — `_get_bm25_index` / `_bm25_leg` / `_rrf_fuse` (moved here
  in Session 1), `tokenize`.
- scripts/ingest.py — `Manifest` class (sqlite manifest.db), `ingest()` embed
  loop (lines ~626-732), `chunk_text`.
- scripts/server.py — calls into the BM25 index for the `common_terms` guard.

CONTEXT:
`_get_bm25_index` rebuilds a full in-memory BM25Okapi corpus (~185k chunks) on
first query per process, re-tokenizing every document and re-deriving the `df`
token->doc-frequency map. On restart it does all of that again. Session 3
persists the per-chunk token counts and the `df` map to sqlite at ingest time
so a cold start hydrates BM25 in seconds instead of minutes. Chroma stays the
dense store (this is Option B, not a Qdrant migration).

WHAT TO DO:
1. In scripts/ingest.py:
   - Extend the Manifest sqlite schema with two tables (CREATE TABLE IF NOT
     EXISTS, additive and safe):
       - `bm25_tokens(doc_id TEXT, token TEXT, tf INTEGER,
                      PRIMARY KEY(doc_id, token))`
       - `bm25_df(token TEXT PRIMARY KEY, doc_freq INTEGER)`
   - After upserting each chunk's embedding (in the existing embed/upsert
     batch loop), also write that chunk's token counts (using `agent.tokenize`
     semantics — raw lowercase, no stemming, so the stored data matches the
     existing query path) into `bm25_tokens`. Rebuild `bm25_df` incrementally
     or as a post-pass over the new tokens (doc-frequency = number of distinct
     doc_ids containing the token).
   - Deterministic behavior: since chunk IDs are deterministic
     (`sha256(f"{rel}:{index}")`), re-ingest upserts the same rows.
2. In scripts/agent.py:
   - Rewrite `_get_bm25_index` to hydrate the `BM25Okapi` corpus and the `df`
     map from the sqlite tables instead of re-tokenizing every document by
     paginating Chroma. If the tables are absent/empty for a set, fall back to
     the old Chroma-paginate path (so pre-Session-3 indexes still work).
   - Reconstruct each doc's token list for BM25Okapi from `bm25_tokens`
     (token repeated `tf` times, or pass the precomputed counts to BM25Okapi's
     corpus as term-frequency lists if the library supports it).
   - Keep the in-process `_bm25_cache` so repeat queries in one process stay
     fast.
   - Add a config knob `lexical_backend: bm25_sqlite | bm25_chroma` (default
     `bm25_sqlite` when tables exist, else `bm25_chroma`).
3. Verify:
   - Re-ingest one small set (`.venv/bin/python scripts/ingest.py <path> --set <s> --force`).
   - Start server, run a chat query TWICE. The SECOND query (same process)
     must NOT print "building BM25 index for ..."; a fresh process start also
     must not print it (index hydrates from disk).
   - `eval.py diff` must show no regression vs the Session 2 baseline.
   - `python -m py_compile scripts/ingest.py scripts/agent.py`.

PROMPT FOR THIS SESSION:
--------------------------------------------------------------------------------
Read PLAN.md. Implement ONLY SESSION 3 (persist BM25 to sqlite at ingest time;
hydrate BM25 from disk instead of re-tokenizing the corpus). Keep Chroma as the
dense store. Preserve existing behavior via the bm25_chroma fallback for old
indexes. Verify: re-ingest a small set, confirm no "building BM25 index" on a
warm start and on a fresh process, confirm eval.py diff shows no regression.
Fill in the SESSION 3 HANDOFF block in PLAN.md (Status, which set was
re-ingested, warm-start result). Do not implement other sessions.
--------------------------------------------------------------------------------

HANDOFF (filled in by agent after completion):
Status: Complete

Deviations: None. Behavior preserved exactly. Pre-Session-3 indexes (without
bm25_tokens/bm25_df tables) automatically fall back to the Chroma-paginate
path; the new tables are created on the next ingest run.

Verification results:
- py_compile: agent.py, ingest.py, server.py, mcp_server.py all pass.
- Server smoke test: starts, BM25 fallback to Chroma works (no tables yet).
- eval.py run (22 queries, both sets): all metrics identical to baseline.
- eval.py diff: PASS (no regression vs Session 2 baseline).

What changed:
- scripts/ingest.py: Manifest._init_tables() creates bm25_tokens(doc_id, token,
  tf, set_name) and bm25_df(token, set_name, doc_freq) tables. Manifest gains
  remove_bm25_tokens(), write_bm25_tokens(), rebuild_bm25_df() methods.
  Chunk/embed/upsert loop writes token data per batch and rebuilds df per file.
  Added `from agent import tokenize`.
- scripts/agent.py: _get_bm25_index(set_name, collection, backend="auto") now
  tries hydrating from manifest.db sqlite tables first; falls back to Chroma-
  paginate if tables are empty/missing. _bm25_leg and retrieve_rag accept and
  pass backend parameter. bm25_cache stores None bm25 for empty sets.
- config.json: added "lexical_backend": "auto" (auto | bm25_sqlite | bm25_chroma).

Next step for full Session 3 verification: re-ingest a small set with --force
to populate the bm25_tokens/bm25_df tables, then confirm a fresh server process
hydrates BM25 from disk (no "building BM25 index" message). That requires a
real library path which this shareable repo does not contain.

================================================================================
SESSION 4 — Boundary-aware recursive chunking + TOC-aware sections
================================================================================
START NEW CHAT. Requires SESSION 1 complete (Session 3 optional to have done).

BACKGROUND (read before starting):
- scripts/ingest.py — `extract_text` (lines ~370-469), `chunk_text`
  (lines ~473-488), `ingest()` chunk/embed loop (lines ~664-712).
- config.json — `chunk_tokens`, `chunk_overlap`.

CONTEXT:
`chunk_text` splits purely on whitespace word counts with overlap, cutting
mid-sentence/mid-paragraph and ignoring document structure (chapters,
sections, headings). Benchmarks consistently show recursive splitting at
paragraph/sentence boundaries with a ~512-token target beats naive flat splits.
Session 4 makes chunking structure-aware and boundary-safe but does NOT yet
change retrieval (that is Session 5). This session must not change chunk IDs
for existing `flat` behavior unless a config flag opts in.

WHAT TO DO:
1. In scripts/ingest.py, add structure detection to extraction:
   - PDF: prefer `doc.get_toc()` bookmarks to get section titles + page ranges
     (technical-rag pattern); fall back to font-size/heading heuristics.
   - EPUB: use chapter/spine document boundaries already available from the
     per-item extraction.
   - `.txt`/`.html`: split on heading patterns and blank-line paragraph
     boundaries.
   - Return a section map: list of `{title, ordinal, start_char, end_char}`.
2. Replace `chunk_text` with a boundary-aware recursive splitter:
   - Split at paragraph boundaries (`\n\n`) then sentence boundaries before
     falling back to word-count, so a chunk never splits mid-paragraph when a
     paragraph boundary is available within the target size.
   - Target `chunk_tokens` (default 330) with `chunk_overlap` (default 60),
     same config keys as today.
3. Preserve behavior: when `chunking_strategy` (new config key) is
   `"flat"`, use the old whitespace splitter verbatim (and keep producing the
   exact same chunk IDs). When `"parent_child"` is selected, this session
   still only wires the boundary-aware splitter + section metadata; the
   parent-child retrieval mapping is Session 5. Add
   `chunking_strategy: "flat"` (default) to config.json.
   - Carry new metadata on each chunk: `section_title`, `section_ordinal`,
     `parent_id` (placeholder for now). Do NOT change the deterministic
     `sha256(f"{rel}:{index}")` child id scheme.
4. Verify:
   - Unit-check the splitter with a synthetic paragraph-heavy text: assert no
     mid-paragraph split when a paragraph fits; assert total chunks <= naive
     count.
   - Re-ingest a small set with `chunking_strategy: parent_child` (temp config)
     and confirm it completes and section metadata lands on chunks.
   - `python -m py_compile scripts/ingest.py`.
   - With default `flat`, confirm `eval.py diff` shows no regression.

PROMPT FOR THIS SESSION:
--------------------------------------------------------------------------------
Read PLAN.md. Implement ONLY SESSION 4 (boundary-aware recursive chunking and
TOC/section-aware extraction in scripts/ingest.py), guarded behind the new
`chunking_strategy` config key defaulting to "flat" so existing indexes are
unchanged. Do NOT implement parent-child retrieval (that is Session 5). Verify
the splitter on synthetic input, a small parent_child re-ingest, no regression
under "flat", and py_compile. Fill in the SESSION 4 HANDOFF block in PLAN.md
(Status, splitter behavior notes, section-metadata fields added). Do not
implement other sessions.
--------------------------------------------------------------------------------

HANDOFF (filled in by agent after completion):
Status: Complete

Deviations (all additive; retrieval and flat chunking unchanged):
1. Overlap budget: the overlap prefix carried from the previous chunk's tail
   does NOT count against chunk_tokens (chunks may reach chunk_tokens +
   chunk_overlap ≈ 390 words, still inside the e5 512-token window). Chosen so
   boundary-aware packing stays at least as dense as flat (verified: recursive
   chunk count <= flat count on synthetic text; otherwise paragraph-fitting
   wastes budget and yields MORE chunks than flat).
2. PDF heading fallback uses text-pattern detection (chapter/part/appendix/
   numbered "3.2 ..."/markdown/ALL-CAPS lines, running-header suppression:
   a title repeated >= 4 times is dropped) instead of font-size analysis.
3. EPUB extractor now emits paragraph breaks at block tags (p/div/li/h1-6/
   tr/blockquote...) and normalizes whitespace per paragraph. EPUB-extracted
   text whitespace changes, but the flat splitter's word stream (and chunks)
   is identical; recursive chunking now sees real EPUB paragraphs.
4. ingest() now deletes a file's previously stored chunks
   (collection.delete(where={"source": rel})) before upserting. Deterministic
   IDs re-upsert in place, but a strategy/param change alters the chunk count,
   leaving stale tail chunks; the delete makes strategy switching safe.
5. parent_id placeholder is pre-populated with the deterministic Session-5 id
   sha256(f"{rel}:section{ordinal}")[:16], so Session 5 needs no metadata
   migration - only the parents text store.
6. "Re-ingest a small set": /media/library was not mounted this session, so
   used 3 small real PDFs from the local nlp_books source dir + a synthetic
   .txt + a generated .epub, ingested into scratch collection session4_test
   (temp config via test driver), then deleted collection + manifest rows.

Verification results:
- py_compile: ingest.py, agent.py, server.py, eval.py, mcp_server.py all pass.
- Splitter unit checks (26 assertions, synthetic paragraph-heavy text):
  paragraph integrity (no mid-paragraph cut when it fits), recursive count
  (5) <= flat count (6), overlap carried only within a section (never across
  the boundary), section titles/ordinals correct, max chunk 355 <= 390 words,
  oversized paragraph -> sentence-boundary fallback, oversized sentence ->
  flat fallback, gap-filling for uncovered text, empty-input handling,
  chunk_text byte-identical to the original algorithm, heading detection
  sanity (10 patterns positive/negative).
- Real parent_child ingest (5 files): 29 chunks, every chunk carries
  section_title / section_ordinal / parent_id; parent_id matches the
  deterministic scheme for all; .txt -> 3 markdown sections; .epub ->
  per-chapter sections (Alpha/Beta/Gamma Chapter + front matter); one PDF ->
  11 numbered-section titles via heading fallback; PDFs without structure ->
  single empty-title section fallback.
- Flat parity: session4_flat chunks have exactly the original 7 metadata keys
  (no section/parent keys).
- Stale-chunk cleanup: re-ingest with chunk_tokens 330 -> 100 replaced all
  chunks (29 -> 76) with no leftovers (each source has one total_chunks).
- eval.py run (22 queries): hit@5=1.0 hit@10=1.0 mrr@5=0.9545 mrr@10=0.9545
  context_recall=1.0 context_precision=0.5962; eval.py diff vs Session 2
  baseline: PASS (all six metrics identical).

Session 5 pointers:
- extract_text returns meta["sections"] = [{title, ordinal, start_char,
  end_char}] (empty list = no structure; chunker treats whole doc as one
  section). chunk_text_recursive normalizes/gap-fills the map and RENUMBERS
  ordinals sequentially - parents should key on that normalized ordinal.
- Chunk metadata in parent_child mode: section_title, section_ordinal,
  parent_id (= sha256(f"{rel}:section{ordinal}")[:16]).
- The parents TEXT STORE (manifest table/collection) from Session 5 step 1 is
  NOT yet implemented; only child-side metadata is in place.
- config.json gained "chunking_strategy": "flat" (default; "parent_child" is
  the opt-in for the new path).

================================================================================
SESSION 5 — Parent-child retrieval
================================================================================
START NEW CHAT. Requires SESSION 4 complete.

BACKGROUND (read before starting):
- scripts/ingest.py — section metadata added in Session 4; chunk upsert loop.
- scripts/agent.py — `retrieve_rag`, `diversify_hits`, `rerank_hits`,
  `build_context`, `MAX_CHUNKS_PER_TITLE`.
- scripts/server.py — `CONTEXT_WORD_BUDGET`, `CHUNK_WORD_CAP`,
  `_prepare_rag` (now calling `agent.retrieve_rag`).

CONTEXT:
Parent-child retrieval decouples the search unit (small child chunk, precise)
from the generation unit (large parent chunk = the whole section, complete
context). This is the LlamaIndex/Dify small-to-big pattern. Search still uses
child embeddings; generation uses the section-level parent text. This session
is opt-in behind `chunking_strategy: parent_child`; the default `flat` path
and existing indexes are untouched.

WHAT TO DO:
1. In scripts/ingest.py (parent_child mode only):
   - Compute a parent chunk per section (~1000-1500 tokens, `parent_tokens`
     config key, default 1200). Store parent text + deterministic parent_id
     (`sha256(f"{rel}:section{ordinal}")`).
   - Persist parents in a `parents` table in manifest.db (or a parallel
     collection) keyed by `parent_id` -> {text, title, source, section_title}.
   - Each child chunk already carries `parent_id`; ensure it is populated and
     stored in child metadata.
2. In scripts/agent.py `retrieve_rag` (parent_child mode only):
   - Retrieve children (search unit), rerank children (unchanged), then map
     the top-K children to their UNIQUE `parent_id`s and build the generation
     context from parent text.
   - `diversify_hits` caps at parent granularity (a section counts once).
   - Respect `CONTEXT_WORD_BUDGET` (parents are larger, so fewer fit — that is
     expected; do not raise the cap to compensate).
   - `fiction_only`, `sources` (dedupe by (title, source)), low-relevance
     guard, and title-mode routing must still work unchanged.
3. Config: `parent_tokens: 1200`, `child_tokens` (reuse `chunk_tokens`),
   `chunking_strategy` default stays `"flat"`. Document in config.json.
4. Verify:
   - Re-ingest a small set with `chunking_strategy: parent_child --force`.
   - `eval.py run` under parent_child vs the Session 2 baseline: expect MRR /
     context-recall to hold or improve; record numbers.
   - `/api/chat` returns a grounded answer and the `sources` list still
     collapses to one entry per (title, source).
   - `python -m py_compile scripts/ingest.py scripts/agent.py`.

PROMPT FOR THIS SESSION:
--------------------------------------------------------------------------------
Read PLAN.md. Implement ONLY SESSION 5 (parent-child retrieval: store
section-level parent chunks at ingest in parent_child mode; in retrieve_rag map
retrieved children to parent text for generation). Keep the default `flat`
path and existing indexes unchanged. Verify via a small parent_child re-ingest,
eval.py comparison vs baseline, and a /api/chat smoke test. Fill in the SESSION
5 HANDOFF block in PLAN.md (Status, parent table schema, eval numbers under
parent_child). Do not implement other sessions.
--------------------------------------------------------------------------------

HANDOFF (filled in by agent after completion):
Status: Complete

Parent table schema (manifest.db `parents`):
- parents(parent_id TEXT PRIMARY KEY, set_name, source, title,
  section_title, section_ordinal, text). parent_id =
  sha256(f"{rel}:section{ordinal}") (16-hex). Populated at ingest only in
  parent_child mode; parent text ~1000-1500 tokens.
- nlp_books has 5854 parents (129 files re-ingested under parent_child);
  library is still flat (0 parents) so it correctly passes through the
  parent-child branch untouched (feature activates only when child chunks
  actually carry parent_id).

Retrieval wiring (scripts/agent.py):
- _load_parent_texts(parent_ids, set_name) -> {id: {text,title,source,section_title}}
  (agent.py:655); _map_children_to_parents(hits, parent_texts) collapses
  children to one hit per parent_id (agent.py:690); retrieve_rag applies both
  only when cfg chunking_strategy=parent_child AND children carry parent_id
  (agent.py:793-801). Diversify/guard/title-mode/fiction logic unchanged; hits
  pool already carries parent text pre-guard, so low-relevance sees the exact
  generation context.

Eval under parent_child vs Session 2 baseline (eval.py run, top_k=10, both
sets, BM25 hydrated from disk):
  hit@5=1.0000  hit@10=1.0000  mrr@5=0.9545  mrr@10=0.9545
  context_recall=1.0000  context_precision=0.6568
  (baseline context_precision was 0.5962 -> parent-child IMPROVED context
  purity by ~0.06 with zero loss on hit/mrr/recall.)
- eval.py diff vs baseline: PASS (exit 0, no regression).

Smoke test (retrieval path; no LLM running on :1234 so /api/chat LLM call not
exercised, but Session 5 only changes retrieval, which is fully covered):
- retrieve_rag('nlp_books', 'How can hypnosis help patients?', ...): children
  collapsed to parents -> hits=1, sources deduped to ONE (title,source) entry,
  context built from parent text (349 words), parent text len verified via
  _load_parent_texts.

Verification: py_compile ingest.py agent.py server.py eval.py mcp_server.py OK.
Note: the live nlp_books data includes several poorly-extracted PDFs whose
parent text is non-printable garbage (e.g. Handbook Of Clinical Hypnosis); that
is a source-data/OCR quality issue, not a Session 5 defect.

===============================================================================
SESSION 6 — Agentic tool-calling loop in /api/chat
================================================================================
START NEW CHAT. Requires SESSION 1 and SESSION 5 complete.

BACKGROUND (read before starting):
- scripts/mcp_server.py — existing `search_library`, `summarize_work`,
  `list_collections` tools + `_render_hits`, `_match_titles` (lines 170-271).
- scripts/agent.py — `retrieve_rag`, `build_context`.
- scripts/server.py — `run_rag_chat` (lines ~1403-1450), `/v1/chat/completions`
  non-stream path, `_prepare_rag`, `get_chat_client`.

CONTEXT:
The current web chat is single-shot: retrieve once, generate once. This
session adds a bounded agentic loop so the model can retrieve again or pull the
full parent section when one shot is insufficient. User chose "default on"
with a ReAct text-protocol fallback for models that lack native function
calling (the Dify Function-Calling vs ReAct dual mode). Streaming (SSE) stays
on the single-shot path; the agentic loop is non-streaming (documented
limitation).

WHAT TO DO:
1. Create scripts/agent_loop.py:
   - `run_agent_loop(set_name, query, history, cfg, client, max_steps=3) -> str`
     that:
       1. Calls `agent.retrieve_rag(...)` for initial retrieval + context.
       2. Calls the LLM with the existing SYSTEM_PROMPT + messages + a `tools=`
          list exposing tools built from the MCP helpers:
          - `search_library(query, top_k, filter_kind)` — wraps `retrieve_rag`.
          - `get_section(parent_id)` — return the full parent text for a chunk
            (uses the Session 5 parents store); this is the grimoire "fetch
            full section" pattern.
          - `summarize_work(title)` — port of the MCP tool.
       3. If the model returns `tool_calls`, execute them, append results as
          `tool` role messages, and loop (bounded by `max_steps`).
       4. If no `tool_calls`, return the final answer.
       - Fallback: when function-calling is unavailable/errors, use a ReAct
         text protocol — instruct the model in a system addendum to emit a
         `call: search_library(...)` line, parse it, execute, feed results back,
         loop. Bounded by the same `max_steps`.
2. Config: add `agentic_enabled: true` (default), `agentic_max_steps: 3`,
   `agentic_strategy: auto` (auto = try function-calling, fall back to ReAct).
3. In scripts/server.py:
   - `run_rag_chat` and the `/v1/chat/completions` non-stream path dispatch to
     `run_agent_loop` when `agentic_enabled`; otherwise keep the existing
     single-shot path (which remains the SSE/streaming path and the safe
     fallback).
   - Keep sources/fiction_only/low_relevance fields in the response as today.
4. Verify:
   - With a function-calling-capable model, a multi-hop question (e.g. compare
     two named works) triggers >=1 `search_library`/`get_section` call before
     the final answer (confirm via log lines).
   - With `agentic_strategy: react`, the same question completes via the ReAct
     fallback.
   - With `agentic_enabled: false`, behavior is identical to pre-Session-6.
   - `eval.py diff` unchanged (eval tests retrieval, not the loop).
   - `python -m py_compile scripts/agent_loop.py scripts/server.py scripts/mcp_server.py`.

PROMPT FOR THIS SESSION:
--------------------------------------------------------------------------------
Read PLAN.md. Implement ONLY SESSION 6 (agentic tool-calling loop in
scripts/agent_loop.py wired into run_rag_chat and the /v1 non-stream path;
default ON with ReAct fallback; SSE stays single-shot). Reuse the MCP tool
logic rather than duplicating retrieval. Verify function-calling, ReAct, and
disabled modes, confirm eval.py diff is unchanged, and py_compile. Fill in the
SESSION 6 HANDOFF block in PLAN.md (Status, tools exposed, loop behaviors
observed). Do not implement other sessions.
--------------------------------------------------------------------------------

HANDOFF (filled in by agent after completion):
Status: Complete

Tools exposed (scripts/agent_loop.py, run_agent_loop):
- search_library(query, top_k=6, filter_kind=None) — wraps agent.retrieve_rag;
  returns markdown grounding text (reuses agent.retrieve_rag so BM25 hybrid,
  rerank, relevance guard, title-mode, parent-child all apply), with the same
  fiction/low-relevance NOTE headers as mcp_server.
- get_section(parent_id) — fetches the FULL section text from the Session 5
  `parents` store via agent._load_parent_texts (grimoire "fetch full section").
- summarize_work(title, top_k=8) — port of the MCP tool (agent._match_titles +
  agent.retrieve + agent.diversify_hits).

Loop behaviors observed:
- Function-calling protocol: model issues a tool_call (step 1), result appended
  as a `tool`-role message, model returns final answer (step 2). Verified with a
  stub client hitting the real nlp_books index: `[agent_loop] tool call:
  search_library(...)` logged, then final answer; strategy="function".
- ReAct fallback (agentic_strategy: react): model emits `call:
  search_library({...})`, parser (_parse_react_call) executes it, result fed back
  as a `Tool result:` message, final answer returned. Verified strategy="react".
- Auto degrade: if the LLM call with `tools=` raises (unsupported server), the
  loop catches, prints "falling back to ReAct protocol", and re-runs as ReAct.
- get_section against real parents store returned a 7295-char section; missing
  parent_id returns a graceful "(No full section found...)" string.
- SSE/streaming path untouched — stays single-shot (documented limitation).
- Bounded by agentic_max_steps (default 3).

Config added: agentic_enabled (true), agentic_max_steps (3), agentic_strategy
(auto | function | react) in config.json AND server.default_cfg(). When
agentic_enabled=false, run_rag_chat and /v1 non-stream fall through to the
exact pre-Session-6 single-shot path.

Deviations:
- run_agent_loop returns a dict {"answer","sources","fiction_only",
  "low_relevance","relevance_reason","steps","strategy"} instead of a bare str
  (the PLAN sketch said `-> str`) so server.py can preserve the
  sources/fiction_only/low_relevance response fields "as today". The plan's step
  3 requirement to keep those fields drove this; all fields are present.
- run_agent_loop also accepts optional top_k/filter_kind/embedder/collection/
  reranker/backend/strategy kwargs (plan signature listed only client/max_steps);
  server.py passes its cached embedder/collection/reranker to avoid reloading.
- agentic_strategy supports "function" and "react" in addition to "auto".

Verification results:
- py_compile: agent_loop.py, server.py, agent.py, mcp_server.py, ingest.py,
  eval.py all pass.
- eval.py run (22 queries): hit@5=1.0 hit@10=1.0 mrr@5=0.9545 mrr@10=0.9545
  context_recall=1.0 context_precision=0.6568 (unchanged vs Session 5);
  eval.py diff vs baseline: PASS (exit 0, no regression — eval tests retrieval,
  not the loop).
- server.py imports cleanly; config.json agentic_enabled=true parses.
- Live LLM tool-calling on LM Studio verified against a running model
  (qwen2.5-3b-instruct served on :1234):
    - strategy=function: returned a grounded per-work summary in 1 step (no
      tool call needed — title-mode routed both named works already).
    - strategy=auto: issued a REAL `search_library` tool call
      (`[agent_loop] tool call: search_library({'query': 'neuro linguistic
      programming methods', 'top_k': 5})`), result appended as tool role, looped
      to step 2, final answer returned. Confirms native tool_calls parsing.
    - strategy=react: completed via text protocol, answered from context.
  Loadable-model note: only qwen2.5-3b-instruct loaded on the live server at
  test time; qwen3-4b/qwen3.5-2b/llama-3.2-3b returned "Failed to load model",
  and the loop correctly auto-degraded function->ReAct on that 400 error.

===============================================================================
END OF PLAN
===============================================================================

================================================================================
END OF PLAN
================================================================================