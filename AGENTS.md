# AGENTS.md

## Plan File Format

When a user asks to "write a plan" or "create a plan file", create a new `.md`
file named after the feature or topic (e.g., `myfeature-plan.md`) in `./plans`.
`./plans/` is gitignored — plans are working notes and are never pushed to the
repo. Write the plan using the following structure and formatting conventions.
Do not begin implementing the plan: only create the file.

### File structure

1. **Header block** — 80-`=` divider, then:
   - `REPO NAME — PLAN TITLE` (ALL CAPS)
   - `Repo:`, `Venv:`, `Run tests:` lines
   - `Dev server log:` line (omit if inapplicable)
   - `HOW TO USE THIS FILE` section — short prose explaining the session-based
     workflow, then a `--------------------` (80 `-`) dash fence
   - `KNOWN PRE-EXISTING TEST FAILURES` list (if any are known)

2. **Scope overview block** — 80-`=` divider, then:
   - `SCOPE OVERVIEW — N SESSIONS` heading
   - Numbered one-line summary per session

3. **One SESSION block per logical unit of work**, each containing:
   - 80-`=` divider
   - `SESSION N — Short title` heading
   - 80-`=` divider (repeated)
   - `START NEW CHAT` / dependency note — e.g. "START NEW CHAT for this
     session." or "START NEW CHAT. Requires SESSION 1 complete."  Each
     session is one chat by default, but multiple small independent
     sessions (e.g. a simple 2-line template change) can be grouped into
     a single chat if the prompt notes it.
   - `BACKGROUND (read before starting):` — file and repo paths to read
   - `CONTEXT:` — prose explaining current state and rationale (omit for
     trivial sessions)
   - `WHAT TO DO:` — numbered steps with sub-steps indented 2 spaces, using
     `-` for sub-bullets
   - `PROMPT FOR THIS SESSION:` — verbatim prompt block wrapped in 80-`-`
     lines (the prompt should tell the agent to read this plan file, implement
     the session, run tests, and write a HANDOFF block)
   - `HANDOFF (filled in by agent after completion):` — initially left blank
     with a `Status: TBD` placeholder; the agent fills in migration numbers,
     test results, deviations
   - 80-`=` divider

### Formatting rules

- One blank line between sections inside a session block.
- Labels: `BACKGROUND (read before starting):`, `CONTEXT:`, `WHAT TO DO:`,
  `PROMPT FOR THIS SESSION:`, `HANDOFF (filled in by agent after completion):`.
- Dividers: 80 `=`; prompt fences: 80 `-`.
- Sub-step bullets: 2-space indent + `-`.
- HANDOFF block goes last, starts with `Status: TBD`. Fill after completion.
- End with `END OF PLAN` in 80 `=`. Same fence as dividers.

## Project conventions

### Reading this repo

- `scripts/` — all Python, one module per concern (Flask API, retrieval,
  quiz pipeline, OCR, eval, MCP). Scripts import each other as modules and run
  in-place from `scripts/` (not installed as a package).
- `web/index.html` — single-page UI (vanilla JS + vendored
  `marked.min.js`/`purify.min.js` in `web/vendor/`); talks to the Flask API in
  `scripts/server.py`.
- Docs: `README.md` (user docs), `details.md` (RAG chain walkthrough),
  `quiz.md` (quiz pipeline), `tuning.md` (parameter guidance), `issues.md`
  (known issue statuses). Read the relevant one before touching that area.
- `./plans/improvements.md` — handoff review of known issues (S/P/R/U
  findings, with file:line references). Read before large changes. Note this
  file lives in `./plans/` and is gitignored.

### Running code

- The Python venv lives at `.venv/` (created by `run.sh` via `uv` on the
  system `python3`).
- Dependencies come from `pyproject.toml` (setuptools, core + extras), not the
  old requirement files. `requirements.txt`/`requirements-gpu.txt` are kept only
  as reproducibility locks. Core install: `uv pip install -e .` (gets CPU torch
  from PyPI on Linux). Optional feature extras: `-e ".[rapidocr]"`, `.[anki]`,
  `.[gpu]`; `run.sh` also accepts `RAG_PIP_EXTRAS=rapidocr,anki`. GPU CUDA torch
  is resolved by `run.sh` via the PyTorch index (extra can't express the URL).
- Embedding device is auto-detected: `scripts/_paths.py` `resolve_device()`
  returns `cuda` when `torch.cuda.is_available()` else `cpu`, honoring the
  `embed_device` config (`auto` | `cpu` | `cuda`). Never hardcode
  `CUDA_VISIBLE_DEVICES=""` to force CPU in app code; that blocks the GPU path.
- Always run scripts with `.venv/bin/python scripts/<name>.py`.
- Launchers: `./run.sh` (web app, `scripts/server.py`, default
  http://127.0.0.1:5000) and `./run_mcp.sh` (MCP server, stdio or `--http`).
- Every script derives its data locations from `scripts/_paths.py`
  `rag_root()`: by default the repo root, overridden by the `RAG_ROOT` env var
  for symlinked deployments where data lives outside the code.

### Config

- `config.json` is the git-tracked default config; machine-specific overrides
  and secrets live in the gitignored `config.local.json`, merged over defaults
  via `_paths.merge_local_config`. `llm_api_key` is the only `SECRET_KEYS`
  entry — never persist it into `config.json` or return it to the client.

### Verification (no unit tests)

- There is no pytest/unit-test suite. Compile checks (per `tuning.md`):
  - `python -m py_compile scripts/*.py`
  - `node --check` on the extracted inline JS from `web/index.html`
- The regression gate is the offline eval harness `scripts/eval.py`
  (deterministic, no LLM calls), scored against `evals/golden.jsonl`:
  - `scripts/eval.py run` — evaluate retrieval/topic metrics → `evals/results.json`
  - `scripts/eval.py baseline` — snapshot results as `evals/baseline.json`
  - `scripts/eval.py diff` — compare results vs baseline; exits nonzero on
    regression beyond `--tolerance` (CI gate)
- After changing retrieval/embedding code, run `run` then `diff` to prove no
  regression. Requires an ingested index for the golden set's collections.