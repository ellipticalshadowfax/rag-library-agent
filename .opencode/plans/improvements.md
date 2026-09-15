# Improvements — Handoff

Review of the RAG library agent codebase. This file is a handoff for future
implementation; **no code changes have been made**. Findings are grouped by
category with severity, `file:line`, description, and a suggested fix.

Line numbers refer to the repo as reviewed (Sep 2026).

---

## Security

### S1 — No authentication on any route [High]
- **Location:** `scripts/server.py:37, 533-1759`
- Every `/api/*` and `/v1/*` route is unauthenticated. `CORS(app)` allows all
  origins. Anyone who can reach the port can run scans/ingests/OCR, burn LLM
  budget, read conversations, read/modify config, and enumerate the filesystem.
- **Risk if exposed beyond loopback:** full control of indexing, LLM budget
  abuse, filesystem reads. `run.sh:19` defaults to `127.0.0.1`; `RAG_HOST=0.0.0.0`
  exposes it with zero auth and plaintext HTTP (dev server, no TLS).
- **Fix (deferred by decision):** add a token/session gate to all routes,
  restrict CORS to specific origins, warn loudly if `RAG_HOST` is non-loopback.
  Decision: skip auth for now; document that exposure is trusted-LAN only.

### S2 — `llm_api_key` stored/returned in plaintext, git-tracked [High]
- **Location:** `scripts/server.py:86-89, 545, 785-787, 935-936`; `.gitignore`
- `save_cfg` writes the key into `config.json`, which **is git-tracked**
  (`.gitignore` ignores `config.local.json` but NOT `config.json`). The key is
  also returned to the client by `GET /api/config` and the apply/setup responses.
- **Fix:** move secrets to env var or a gitignored file (`config.local.json`);
  mask `llm_api_key` in all JSON responses.

### S3 — `cid` path construction not validated [Med]
- **Location:** `scripts/chat_store.py:20-21`; `scripts/server.py:1544-1574`
- `_path(cid)` builds `CONV_DIR / f"{cid}.json"` from raw input. Note: Werkzeug's
  default `string` converter rejects `/`, so `%2F` traversal is likely not
  reachable today — but it is a fragility/defense-in-depth gap (a future
  `<path:>` route or server encoding quirk would expose arbitrary `.json` reads
  and deletes).
- **Fix:** whitelist `^[0-9a-f]{12}$` (matching `uuid.uuid4().hex[:12]`).

### S4 — SSRF via `llm_base_url` + key exfiltration [Med]
- **Location:** `scripts/server.py:156-168`
- `llm_base_url` is user-controlled and fetched server-side, sending
  `llm_api_key` as the `Authorization` header. An attacker can point it at
  internal hosts / the cloud metadata IP (`169.254.169.254`).
- **Fix:** validate/restrict `llm_base_url` to localhost or an allowlist.

### S5 — No `MAX_CONTENT_LENGTH`, no rate limiting, unvalidated `top_k` [Med]
- **Location:** `scripts/server.py:550, 1260-1407`
- `get_json(force=True)` accepts unbounded bodies. No throttling on expensive
  chat endpoints (embedding + LLM + rerank per request). `int(data.get("top_k"))`
  crashes on non-integers and accepts huge values.
- **Fix:** set `MAX_CONTENT_LENGTH`; add per-IP throttling; clamp/validate `top_k`.

### S6 — `/api/fs` and `/api/embed/load` [Low]
- **Location:** `scripts/server.py:1722-1745, 713-726`
- `/api/fs` lists any directory on the host (info disclosure). `/api/embed/load`
  downloads arbitrary HF models (also a DoS vector).
- **Fix:** restrict `/api/fs` to configured library roots; restrict models to the
  curated `EMBED_MODELS` list.

---

## Performance

### P1 — BM25 index build blocks first request and is unlocked [High]
- **Location:** `scripts/agent.py:511-517, 576-603`
- `_bm25_cache` builds inline on the first chat request (paging ~186k chunks +
  tokenizing every chunk) — first-request latency is minutes. It is also unlocked,
  so concurrent first-requests each rebuild it redundantly.
- **Fix:** guard with a per-set `threading.Lock` (double-checked locking); warm in
  a background daemon thread at startup; invalidate after re-ingest.

### P2 — Streamed response holds first token hostage to retrieval [High]
- **Location:** `scripts/server.py:1320-1323`
- `_stream_sse` runs `_prepare_rag` (full retrieval incl. rerank) synchronously
  inside the generator before the first `yield`. SSE headers flush but the first
  token waits on the entire retrieval pass.
- **Fix:** run retrieval before returning the `Response` or in a worker thread;
  emit a "retrieving…" progress event.

### P3 — Reranker scores 40 candidates but keeps 24 [Med]
- **Location:** `scripts/agent.py:411-420`
- `rerank_hits(..., top_n=24)` scores **all** fused candidates (up to 40) on a CPU
  cross-encoder, then trims to 24 — 16 redundant pairs scored per request.
- **Fix:** truncate `pairs` to `top_n` before predicting.

### P4 — Fresh Chroma client per request; unused `_chat_lock` [Med]
- **Location:** `scripts/server.py:484, 495-497, 1101, 1174, 1437`
- `get_chat_collection` constructs a new `PersistentClient` per request and never
  closes it (file-handle/memory leak). `_chat_lock` is declared but never used, so
  embedder/reranker/client lazy-init is unguarded (check-then-act races).
- **Fix:** cache the collection/client per set; wrap lazy-init in `_chat_lock`.

### P5 — Minor per-request redundancy [Low]
- **Location:** `scripts/agent.py:764, 767, 835`; `scripts/server.py:500-528`
- BM25 cache not invalidated after re-ingest; `_match_titles` re-queries
  manifest.db and rescans all titles every request; embedder/reranker load inline
  at first request.
- **Fix:** invalidate BM25 on ingest; cache the title list per set; preload models.

---

## Reliability

### R1 — Stale `.ocr.lock` permanently blocks OCR [High]
- **Location:** `scripts/server.py:377-378, 406-411`
- `ocr_active()`/`read_ocr_status()` treat any lock file (and its PID) as live
  without checking liveness. A killed OCR process leaves a stale lock that blocks
  Start OCR forever. Ingest correctly validates PIDs (`ingest.py:54-62`).
- **Fix:** validate PID liveness (e.g. `os.kill(pid, 0)` + `/proc` start-time) and
  clear stale locks, matching ingest.

### R2 — `_sse_wrap` has no error handling [Med]
- **Location:** `scripts/server.py:1495-1526`
- The OpenAI-compatible streaming endpoint iterates the LLM stream with no
  try/except. If the LLM dies mid-stream, external clients hang with no
  error/`[DONE]`.
- **Fix:** wrap the loop and emit `data: {"error": ...}` + `data: [DONE]`.

### R3 — `_prepare_rag` outside SSE try block [Med]
- **Location:** `scripts/server.py:1323`
- `_prepare_rag` (embedder/client init) runs before the try, so retrieval/embedder
  failure yields an HTTP 500 with no SSE `error` event for the web client.
- **Fix:** move it inside the try, or catch and emit an `error` event.

### R4 — Non-atomic JSON writes [Med]
- **Location:** `scripts/server.py:86-89`; `scripts/chat_store.py:71-123`;
  `scripts/ocr_compare.py:464-532`
- `config.json`, conversation files, and `ocr_compare_report.json` are rewritten in
  place. A crash/disk-full mid-write corrupts them; corrupted `config.json` prevents
  server start.
- **Fix:** write to a temp file and `os.replace()` (atomic on POSIX).

### R5 — Resource cleanup / unlocked shared state [Med]
- **Location:** `scripts/ingest.py:597-620, 1199-1201`; `scripts/server.py:42-77, 1759`
- PDF `doc` handle not closed on extraction exception (no `finally`); Chroma client
  not closed on ingest stop; global state dicts mutated without locks under
  `app.run(threaded=True)`.
- **Fix:** `try/finally: doc.close()`; close Chroma/manifest in `finally`; guard
  state dicts with a lock.

### R6 — Error signaling via `sys.exit` and swallowed exceptions [Low]
- **Location:** `scripts/agent.py:59-61`; `scripts/server.py:148-149, 235-236`
- `setup_chroma` uses `sys.exit(1)` (relies on callers catching `SystemExit`);
  several `except Exception: pass` blocks swallow errors silently.
- **Fix:** raise a custom exception; log before `pass`.

---

## Usability

### U1 — README drift [Med]
- **Location:** `README.md:75, 145-157`
- README says "four tabs" but the UI has five (Setup/Scan/OCR/Ingest/Chat). The OCR
  section never mentions `TESSERACT_BIN`.
- **Fix:** correct tab count; document `TESSERACT_BIN` (resolve via
  `shutil.which("tesseract")` first).

### U2 — Machine-specific / personal data committed [Low]
- **Location:** `config.json:23-25`; `scripts/ocr_compare.py:44-45`; `tuning.md:136`
- `exclude: ["Leo Tolstoy"]` (personal), `DEFAULT_TESS_BIN`/`DEFAULT_TESSDATA`
  (machine paths), and `OCR_SAMPLE_DIR` default are in the shareable copy.
  `config.json:38` `lexical_backend` key is dead code.
- **Fix:** neutralize defaults (`exclude: []`, `shutil.which`), use a neutral
  sample dir, and wire up or remove `lexical_backend`.

### U3 — run.sh uses system `python3`; doc/config drift [Low]
- **Location:** `run.sh:96`; `scripts/server.py:120`
- run.sh's LLM check uses the system `python3` instead of the venv. `max_tokens`
  default is 2048 in code vs 4096 in config/docs.
- **Fix:** use `$HERE/.venv/bin/python`; reconcile defaults.

---

## Decision notes
- **Auth (S1):** deferred — trusted-LAN-only deployment is the current assumption.
- **Scope of this file:** review/handoff only; no code changed.