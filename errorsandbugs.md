# Errors & Bugs — Codebase Review (2026-10-08)

Fresh review of `rag-library-agent`. Sanity gates pass: `python -m py_compile
scripts/*.py` and `node --check` on the extracted inline JS from
`web/index.html` (both clean). Prior findings from `plans/improvements.md`
(Sep handoff) and `issues.md` were re-checked against current code; statuses
are in sections B and C.

Line numbers refer to the repo as of commit `d559718`.

---

## A. New bugs (found this review, not in the prior handoff)

### High severity

#### 1. `/api/chat/stream` hangs forever when the collection is missing
- **Location:** `scripts/server.py:2941-2978`
- `_agent_worker` calls `get_chat_collection(set_name)` (line 2949), which runs
  `agent.setup_chroma` → `sys.exit(1)` (`scripts/agent.py:99-105`) when the
  collection doesn't exist. `SystemExit` is **not** caught by
  `except Exception`, so the worker thread dies without pushing anything on
  `evq`, and the generator's `while True: kind, data = evq.get()` blocks
  forever — a permanently stuck SSE connection + leaked thread per request.
- **Repro:** select a deleted/typo set and ask any non-quiz/non-catalog
  question in agentic mode.
- **Fix:** catch `SystemExit` explicitly in `_agent_worker` (like
  `_quiz_generate_sse` already does at `server.py:1866`), or wrap
  `evq.get()` loop with a sentinel/timeout; better: stop raising `SystemExit`
  in `setup_chroma` (see R6 / bug 13 below).

#### 2. `_stream_sse` single-shot error path raises KeyError
- **Location:** `scripts/server.py:2999-3001`
- `if "error" in payload: text = payload["answer"]` — but `_prepare_rag`'s
  error dict is `{"error": ...}` only (no `"answer"` key, see
  `server.py:2577-2590`). With `chat_mode: single` / `agentic_enabled: false`
  and an empty/missing collection, the generator throws → HTTP 500, no SSE
  `error` event for the client.
- This is a regression of the prior R3 "fix": the call was moved inside the
  generator, but the error branch was written against the wrong key.
- **Fix:** read `payload["error"]` and emit an SSE `error` event.

#### 3. `/api/embed/apply` leaks the plaintext API key
- **Location:** `scripts/server.py:1104-1123`
- Returns `jsonify({"ok": True, "config": cfg, ...})` where
  `cfg = load_cfg()` includes `llm_api_key` merged from `config.local.json`
  (`scripts/_paths.py:43-47`, `scripts/ingest.py:88-92`). Any client that
  POSTs to this route receives the real key. Every other config endpoint
  correctly uses `_public_cfg()` (`server.py:140-146`).
- **Fix:** `"config": _public_cfg(cfg)`.

#### 4. MCP `get_config(keys=[...])` leaks the API key into the chat context
- **Location:** `scripts/mcp_server.py:615-625`
- The `keys` parameter echoes `cfg.get(k)` for **any** requested key,
  including `llm_api_key` (config is loaded merged with local overrides).
  Same class as A3; the default `notable` list excludes the key, the
  `keys` param path does not filter `SECRET_KEYS`.
- **Fix:** refuse keys in `_paths.SECRET_KEYS`.

#### 5. OCR cache is keyed by filename stem only
- **Location:** `scripts/ingest.py:744`, `scripts/ingest.py:1212`,
  `scripts/ocr.py:112`
- All three derive the cache path as `ocr/<stem>.txt`. Two different PDFs
  sharing a stem (e.g. `Chapter 1.pdf` in two folders, or the same stem
  across sets) collide: the second file silently gets the first file's OCR
  text — and in merge mode that text is then written into the second PDF as
  an invisible text layer (`merge_text_into_pdf`), corrupting the source
  file. No hash/mtime validation establishes cache ownership.
- **Fix:** salt the key with the rel path
  (e.g. `f"{stem}-{sha1(rel)[:8]}.txt"`) and/or verify cache freshness vs
  the PDF's pre-merge mtime.

### Medium severity

#### 6. Topic-mode quiz ranking silently never works
- **Location:** `scripts/quiz_plan.py:178`
- `embedder.encode(texts, prompt_name="document")` — the installed
  sentence-transformers 6.1.0 resolves `prompt_name` via
  `self.prompts[prompt_name]` and raises `ValueError` for unknown names
  (verified against the installed package), and the app only registers
  `query`/`passage` prompts (`scripts/agent.py:49-50`). The `except` at
  `quiz_plan.py:184` swallows it and logs once, so `_resolve_topic`
  **always** falls back to inventory order — cosine topic ranking is dead.
- **Fix:** use `prompt_name="passage"`.

#### 7. Course "quizzed" flag is permanently false
- **Location:** `scripts/server.py:2169-2183` (`_course_quizzed_counts`)
- Iterates `q.get("questions")` over entries from `quiz_store.list_quizzes`,
  but that returns summary dicts only (`id/title/set_name/status/
  question_count/created/updated` — no `questions` key;
  `scripts/quiz_store.py:221-229`), so the counts map is always empty and
  `/api/course` unit progress never populates.
- **Fix:** load full quizzes via `quiz_store.get_quiz(qid)` for matching ids,
  or return `questions` in `list_quizzes`.

#### 8. LLM-verify abort discards prior groups' accepted questions
- **Location:** `scripts/quiz_build.py:360-433` (`verify_questions`)
- On any parent group's LLM error: `return {qid: False for qid, _ in items}`
  early-returns the whole function — throwing away `accepted` entries
  accumulated from earlier groups and skipping later groups entirely. The
  whole batch drops ("verify_failed") even though verification worked for
  most parents.
- **Fix:** mark only the failed group's items as `False` and `continue`.

#### 9. GIFT export can't identify the correct MCQ answer
- **Location:** `scripts/quiz_export.py:65` (`render_gift`)
- `key_letter = "".join(ch for ch in answer if ch.isalpha()).upper()` turns
  a full-text answer (`"A) Anna Sewell"`) into `"AANNASEWELL"`, which never
  equals a single letter → no choice gets the `=` marker; Moodle imports
  become ungradable quizzes. Also `true_false` maps only `true`/`t` → TRUE,
  while grading accepts yes/no/1/0 (`scripts/quiz_grade.py:32-34`), so those
  export inverted.
- **Fix:** reuse `quiz_grade._answer_letter` for the letter; extend the TF
  coercion.

#### 10. Rebuild-on-existing-spec duplicates questions
- **Location:** `scripts/quiz_build.py:500-515`, `544-701`
- `build_quiz` resumes by pre-loading existing questions into `accepted`, but
  each unit restarts `unit_accepted = 0` and regenerates the full allocation.
  Only near-identical questions are caught by the dedupe stage, so calling
  generate twice on one spec grows the quiz; `report.per_unit.accepted` also
  undercounts (fresh batch only, ignoring resumed ones).
- **Fix:** seed per-unit accepted counts from the stored questions' `unit`
  tags.

#### 11. `POST /api/config` mask round-trip overwrites the real key
- **Location:** `scripts/server.py:298-397` (`CONFIG_META`),
  `server.py:921-934` (`api_config_set`), `server.py:103-110` (`save_cfg`)
- `llm_api_key` is registered `editable: True` (`server.py:319`), so it is in
  the derived allowlist (`_config_writable_keys`, `server.py:400-406`). Unlike
  `/api/llm/apply` which skips the mask (`server.py:1187`:
  `if key and key != API_KEY_MASK`), `POST /api/config` accepts the masked
  value. A GET-then-POST script/MCP client (the natural "read config, tweak,
  save" pattern) therefore persists the literal `"••••••••"` as the API key
  into `config.local.json`, overwriting the real key.
- **Fix:** strip/skip `API_KEY_MASK` in `api_config_set`, or mark the key
  non-editable there and route writes through `/api/llm/apply` only.

#### 12. llama-server zombie lockout
- **Location:** `scripts/server.py:1267-1290` (`api_llm_start`),
  `server.py:1293-1309` (`api_llm_control`)
- Only `SELFHOST_PROC["pid"]` is kept; the `Popen` object is dropped, so the
  child is never reaped. `os.kill(pid, 0)` succeeds for zombies, so after the
  llama-server process exits, start/control keep reporting "already running"
  until the whole web server restarts. This is the exact bug already fixed
  for OCR/ingest by keeping the Popen and `poll()`-ing it (see the comments
  at `server.py:68-70, 91-94`) — the llama path was never migrated.
- **Fix:** store the `Popen` and use `proc.poll()` for liveness.

#### 13. `/api/chat` (non-stream) dies without a response on missing collection
- **Location:** `scripts/server.py:2668-2689` (`run_rag_chat`)
- `get_chat_collection(set_name)` is evaluated inside the agentic `try`, but
  `except Exception` does not catch the `SystemExit` from `setup_chroma`
  (`scripts/agent.py:105`), so it escapes `run_rag_chat` entirely instead of
  falling back to `_prepare_rag`'s clean `{"error": ...}` path. Client sees a
  dropped connection. Root cause shared with A1/A2: error-signaling via
  `sys.exit` (old finding R6).
- **Fix:** catch `SystemExit` alongside `Exception` in `run_rag_chat` (and
  in the `_stream_sse` worker), or raise a normal exception in
  `setup_chroma`.

#### 14. `extract_count` counts any number in the quiz query
- **Location:** `scripts/catalog.py:131-141`
- Regex `\b(\d{1,3})\b` picks up numbers that are part of the topic, not a
  count: "quiz me on 18th-century seafaring" → 18 questions;
  "the 7 habits of highly effective people" → 7.
- **Fix:** anchor on count phrasing ("10 questions", "of 15", "quiz with 20").

#### 15. Pending quiz-spec hijack on follow-up questions
- **Location:** `scripts/server.py:2410-2462` (`_resolve_pending_spec_delta`)
- While a pending spec exists, **any** message matching
  `\b(build|generate|go ahead|make it now|yes|yep|confirm)\b` (line 2423)
  confirms-and-"builds" the quiz instead of answering the question —
  "What generates ATP?" matches `generate`. The adjustment regex on line
  2431 also includes a bare `\d+`, so any number in a normal question
  triggers revision parsing.
- **Fix:** require short imperative-ish messages (e.g. whole-string match)
  before treating as confirm/revise.

#### 16. Ingest prune leaves BM25 postings behind
- **Location:** `scripts/ingest.py:1167-1182`
- The stale-file prune deletes Chroma chunks + `files`/`parents` rows, but
  not the pruned document's `bm25_tokens` rows (chunk doc ids aren't known
  there), and `bm25_df` isn't rebuilt unless another file happens to change
  (`ingest.py:1360`). Deleted books keep matching the lexical leg until a
  manual `python scripts/ingest.py --rebuild-bm25 --set <name>`.
- **Fix:** delete postings by doc-id prefix mapping (or run
  `rebuild_bm25_df` after pruning) and/or clear the in-memory
  `_bm25_cache` (see B/P5).

### Low severity

#### 17. `agent.py` CLI crashes / path bug
- `/sources` before the first question references `hits` before assignment →
  `NameError` kills the REPL (`scripts/agent.py:1172-1176`; `hits` is only
  bound at 1212).
- `cmd_collections` re-derives the data dir as
  `Path(__file__).resolve().parent.parent` (`scripts/agent.py:1125`) instead
  of `rag_root()` — ignores `RAG_ROOT`, the same class of bug that was
  declared fixed in `issues.md` #5 (missed spot).

#### 18. Web UI: fail-open sanitizer + one unescaped interpolation
- `web/index.html:970` — `mdSanitize` returns the raw marked output when
  `DOMPurify` is undefined (vendor file blocked/corrupted), so LLM markdown
  becomes executable HTML. Prefer fail-closed (escape) there.
- Course view injects `<b>${data.title}</b>` into `innerHTML` without
  escaping (self-XSS only — the title comes from the user's own input echoed
  by `/api/course`); the surrounding unit titles/warnings are escaped.

#### 19. OCR status 500 on malformed lock file
- `scripts/server.py:688-730` — `_read_ocr_lock()` returns raw JSON and
  `ocr_active()`/`read_ocr_status()` call `int(ext.get("pid"))` without a
  default; a partial/garbage `.ocr.lock` (missing `pid`) raises `TypeError` →
  500. The ingest path validates this (`ingest.read_ingest_lock` uses
  `int(data.get("pid", -1))`, `scripts/ingest.py:52-60`); the OCR path
  doesn't.

#### 20. `api_quiz_answer` dead validation
- `scripts/server.py:1987-1990` — `qid = str(data.get("qid"))` makes
  `qid is None` unreachable; a missing qid becomes the string `"None"` and
  yields 404 "Question not found" instead of 400.

#### 21. Doc/config/UI drift (U-family, still live)
- `README.md:99` says "four tabs"; the UI has six: Setup/Scan/OCR/Ingest/
  Chat/Study (`web/index.html:253-258`).
- `config.json:42` still commits `"exclude": ["Leo Tolstoy"]` (personal
  data in the shared config).
- `scripts/ocr_compare.py:45-46` still bakes machine-specific tesseract
  paths (`/tmp/opencode/mamba/envs/tess/...`).
- `lexical_backend` is now honored by catalog/MCP
  (`scripts/catalog.py:364`, `scripts/mcp_server.py:263,725`) — part of U2
  fixed — **but** the web-chat path never passes it:
  `server.py:2596-2601` calls `agent.retrieve_rag(...)` without
  `backend=`, so the main chat pipeline always uses `"auto"`. The config
  knob is half-honored.

#### 22. No cancellation of background workers on client disconnect
- `scripts/server.py:1851-1886` (`_quiz_generate_sse`) and
  `server.py:2875-2930` (map-reduce path): if the SSE client disconnects,
  the worker thread keeps running (LLM budget continues to burn; queue
  grows). The quiz build has no abort hook.

---

## B. Prior findings still live (from `plans/improvements.md`)

- **S1** No auth on any route + open `CORS(app)` (`scripts/server.py:45`) —
  deferred by decision (trusted-LAN only), unchanged.
- **S3** Raw-id path joins with no whitelist: `chat_store._path`
  (`scripts/chat_store.py:20-21`) **and now also** `quiz_store._spec_path` /
  `_quiz_path` (`scripts/quiz_store.py:38-43`) used from
  `/api/quiz/<quiz_id>`, `/api/quiz/spec/<spec_id>`. Only mitigated by
  Werkzeug's `string` converter rejecting `/`; no `^[0-9a-f]{12}$` check.
- **S4** SSRF via user-controlled `llm_base_url`, key sent as
  `Authorization` (`scripts/server.py:434-485, 1152-1174`).
- **S5** No `MAX_CONTENT_LENGTH` anywhere; `int(data.get("top_k"))` raises
  `ValueError` → 500 on non-integer input
  (`scripts/server.py:2773, 2831, 3080`).
- **S6** `/api/fs` lists any directory on the host
  (`scripts/server.py:3447-3470`); `/api/embed/load` downloads arbitrary HF
  models (`server.py:1081-1094`).
- **P1** BM25 index built inline on first chat request, `_bm25_cache`
  unguarded (double-check locking missing) — concurrent first requests each
  rebuild; minutes of first-token latency
  (`scripts/agent.py:665-773`).
- **P2** Single-shot path runs full retrieval inside the SSE generator
  before the first byte (`scripts/server.py:2999`).
- **P3** Reranker scores all fused candidates, then trims to `top_n`
  (`scripts/agent.py:537-546` — `pairs` built from every hit before
  `predict`).
- **P4** Fresh `chromadb.PersistentClient` per request, never closed
  (`scripts/server.py:862-864` via `setup_chroma`, `server.py:3273`);
  `_chat_lock` declared at `server.py:851` and **never used** → unguarded
  lazy-init races for `_chat_client`/`_chat_embedder`/`_reranker` under
  `threaded=True`.
- **P5** In-memory `_bm25_cache` is never invalidated after ingest; the
  full Chroma-based rebuild is never persisted back to manifest.db
  (`scripts/agent.py:741-768` never calls `replace_bm25`), so a sparse
  persisted index forces a minutes-long rebuild **every process start**;
  `_match_titles` re-queries + rescans all titles every request
  (`scripts/agent.py:244-330`).
- **R2** `_sse_wrap` has no error handling around the LLM stream
  (`scripts/server.py:3207-3220`) — external OpenAI-compatible clients hang
  with no `error`/`[DONE]` if the stream dies mid-response.
- **R4** Non-atomic JSON writes remain for `config.json`
  (`scripts/server.py:113-115` `_write_json`) and every `chat_store` write
  (`scripts/chat_store.py:71,92,105,119,146,172`). (Partially fixed — see
  C.)
- **R5** `merge_text_into_pdf` leaks the fitz `doc` on exception (close()
  inside the try, `scripts/ingest.py:586-604`); shared state dicts
  (`INGEST_STATE`/`OCR_STATE`/`SCAN_STATE`/`_reranker`) mutated without
  locks.
- **R6** `setup_chroma` still signals errors via `sys.exit(1)`
  (`scripts/agent.py:105`) — root enabler of A1/A2-equivalent bugs
  (`server.py:1437`-style `except SystemExit` guards are everywhere, which
  is the fragility R6 predicted).

---

## C. Confirmed fixed since the Sep handoff

- **S2** Secret plumbing: `SECRET_KEYS` + `config.local.json` split, masked
  client values (`scripts/_paths.py:23`, `scripts/server.py:103-146`),
  `config.json` no longer carries a key — except the three leaks
  (A3 embed/apply, A4 MCP get_config, A11 mask round-trip).
- **R1** Stale OCR/ingest lock liveness: `ocr_active()`/`read_ocr_status()`
  now validate PIDs and `_clean_stale_lock()` clears dead locks
  (`scripts/server.py:696-730, 762-781`).
- **R3** (attempted): retrieval moved inside the generator try-structure —
  but the fix introduced the KeyError in A2.
- **Manifest set-awareness** (issues.md #3) and `agent.py` `rag_root()`
  usage (issues.md #5) — both verified in current code (one missed spot:
  CLI `cmd_collections`, A17).
- OCR-merge repeat-reprocessing (issues.md #2) — post-merge stat write-back
  confirmed (`scripts/ingest.py:1226-1238`); the `needs_ocr` "pdftotext
  missing" residual remains (issues.md #2 note).
- `quiz_store` writes are atomic (`scripts/quiz_store.py:58-63`).
- `run.sh` uses the venv Python for installs (U3 fixed;
  `run.sh:105-125`); config/UI `llm_max_tokens` defaults now agree at 2048.
- `stale-chunk prune pass` on ingest (issues.md #4) implemented
  (`scripts/ingest.py:1161-1182`) — see A16 for its residual BM25 gap.

---

## Quick wins (ordered)

1. Fix the SSE error/hang trio: A1 + A2 + B13 — catch `SystemExit` in the
   `_stream_sse` agent worker and `run_rag_chat`, and read
   `payload["error"]` instead of `payload["answer"]`. (Or, properly, make
   `setup_chroma` raise a normal exception — old R6.)
2. `server.py:1123` → `_public_cfg(cfg)`; MCP `get_config` filter
   `SECRET_KEYS` (A3 + A4).
3. Salt the OCR cache key with the rel path (A5).
4. One-word fixes: `prompt_name="document"` → `"passage"` (A6); GIFT
   `key_letter` → reuse `quiz_grade._answer_letter` (A9);
   `verify_questions` early-return → per-group `continue` (A8);
   `_course_quizzed_counts` → `get_quiz` per id (A7).

## Verification performed

- `python -m py_compile scripts/*.py` — clean.
- `node --check` on the extracted inline JS of `web/index.html` — clean.
- sentence-transformers 6.1.0 `_resolve_prompt` behavior confirmed against
  the installed venv (raises `ValueError` for unknown `prompt_name`;
  `agent.setup_embedder` only registers `query`/`passage`).
- `evals/golden.jsonl` checked: all 25 rows carry `expected_title`, so
  `eval.py` has no live KeyError exposure (the `item["expected_title"]`
  hard-access at `eval.py:202` would still bite future topic-only rows).
- Offline eval harness (`scripts/eval.py run/diff`) NOT executed — requires
  an ingested index for the golden set's collections, which this checkout
  doesn't have (`manifest.db` empty, `index/` bare).

---

## D. Resolution log (2026-10-09)

Status of the findings above. High/medium A-items (A1–A16) were fixed in commit
`76e2842`. The low-severity A-items and the live B-items were addressed in the
follow-up pass; entries left untouched are noted with a reason.

**Fixed (A-family, low severity):**
- **A17** `agent.py` CLI: `/sources` no longer `NameError`s (`hits = []`
  initialized before the REPL); `cmd_collections` uses `rag_root()`.
- **A18** Web UI: `mdSanitize` now fails **closed** (escapes) when DOMPurify is
  missing; the course title is escaped before `innerHTML`.
- **A19** `_read_ocr_lock` coerces non-object JSON to `None`; malformed/
  missing `pid` no longer raises `TypeError` → 500.
- **A20** `api_quiz_answer` validates a missing/empty `qid` before coercing to
  `str` (400, not the string `"None"` → 404).
- **A21** Drift: `config.json` `exclude` personal entry removed; `ocr_compare`
  no longer bakes machine-specific tesseract paths (env/PATH only); the
  web-chat path now passes `lexical_backend` to `retrieve_rag`. (README tab
  count was already fixed in `3ba4a19`.)
- **A22** Background SSE workers (agent loop, map-reduce, quiz build) now take
  a `threading.Event`; the generator's `finally` sets it on client disconnect,
  and the callbacks raise `_StreamCancelled` to abort the worker.

**Fixed (B-family):**
- **S3** Raw-id path joins (`chat_store._path`,
  `quiz_store._spec_path`/`_quiz_path`, syllabus slug/hash) now require a
  generated 12-hex id via `_paths.safe_id`; invalid ids map to a sentinel.
- **S5** `MAX_CONTENT_LENGTH` (64 MiB) added; untrusted counts/`top_k` go
  through `_as_int` (clamped, no `ValueError` → 500).
- **P1/P5** BM25 build is double-checked under a lock; a Chroma-built index is
  persisted back with `replace_bm25`; the per-set title list is cached; the
  server invalidates both caches when an ingest starts.
- **P3** `rerank_hits` caps cross-encoded candidates (default `2*top_n`).
- **P4** `_chat_lock` now guards lazy client/embedder/reranker init (double-
  checked).
- **R2** `_sse_wrap` wraps the upstream LLM stream in try/except and always
  emits `[DONE]`.
- **R4** `config*.json` and conversation writes go through
  `_paths.atomic_write_json` (temp file + `os.replace`).
- **R5** `merge_text_into_pdf` closes the fitz document in a `finally`.
- **R6** `setup_chroma` raises `agent.CollectionNotFound` (a normal
  `Exception`) instead of `sys.exit(1)`; all server `except SystemExit` guards
  were widened to catch it (eval too). CLI keeps a clean exit.

**Deliberately left as-is:**
- **S1** (no auth / open CORS) — deferred by decision (trusted-LAN only).
- **S4** (SSRF via `llm_base_url`), **S6** (`/api/fs`, `/api/embed/load`) —
  remain host-trust features; not changed without a product decision.
- **P2** single-shot retrieval still runs before the first SSE byte.
- **R5** (shared `INGEST_STATE`/`OCR_STATE` dict mutations without locks) —
  broad; the concrete fitz leak was fixed, the rest left for a dedicated pass.
- **S2** `agent_loop.setup_chroma` self-init path now raises normally, which
  callers already handle via `except Exception`.

Verification this pass: `python -m py_compile scripts/*.py` clean; `node --check`
on the extracted inline JS clean; targeted functional checks for `safe_id`,
`atomic_write_json`, `ocr_cache_name`, store-path guards, `extract_count`,
`_as_int`, `rerank_hits` capping and cache invalidation all pass. The offline
eval harness was **not** run (no ingested index, as in §Verification above).
