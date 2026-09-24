# Quiz System — Overview

The RAG library agent has a **full 6-session quiz pipeline** that plans, generates, grades, reviews, and exports structured quizzes grounded in your local book collection. Every question cites its source so you can verify claims against the actual text.

## Quick start

1. Open the **Study tab** in the web UI.
2. Pick one of three scope modes:

   | Mode     | How it works                                                        | When to use                               |
   |----------|---------------------------------------------------------------------|-------------------------------------------|
   | `work`   | Plans units from every indexed section of a named book              | Full-book coverage                        |
   | `sections` | Picks explicit unit_ids (e.g. `"3-8"`) from a work's syllabus     | Narrow, targeted review                   |
   | `topic`  | Cosine-ranks books + sections via E5 embeddings                     | Cross-library topic search                |

3. Adjust the spec (question count, difficulty, depth) — then click **Build**.

## Session-by-session breakdown

### Session 2 — Planning (`scripts/quiz_plan.py`)

Transforms a user request into an editable **QuizSpec**:

```jsonc
{
  "id": "fff34e",           // persisted under quizzes/specs/<id>.json
  "mode": "topic|work|sections",
  "depth": "surface|balanced|deep",   // how many units to keep
  "count": 30,                          // total questions requested
  "difficulty": "mixed|easy|medium|hard",
  "units": [                            // numbered options with per-unit allocation
    { "unit_id": "u24", "title": "NLP Mastery", "allocation": 5, "words": 1200 }
  ],
  "warnings": ["Topic matches multiple works"]
}
```

**Scope resolution:**
- **work mode** → deterministic walk of `manifest.db.parents`; no LLM call.
- **sections mode** → picks unit_ids from the cached syllabus (supports ranges like `"3-8"`).
- **topic mode** → ranks relevant books with `catalog.find_books`, then cosine-ranks sections using the E5 embedder. No LLM needed for ranking; only for syllabus generation.

**Budget allocation** distributes questions proportionally to unit word counts with a per-unit floor (default 1). A **weakness-weighted** default bumps weight > 1 for units missed in previous attempts.

**Conversational revision** is handled by `revise_spec()`: include/exclude units, change total count, swap depth or difficulty — re-applies allocation instantly.

### Session 3 — Generation & Audit (`scripts/quiz_build.py`)

Walks each unit's parents in order, sends batches to the LLM, then runs a **4-stage audit** before accepting any question:

| Stage | What it checks                              | Tool        | Deterministic? |
|-------|---------------------------------------------|-------------|----------------|
| 0     | Parse sanity: valid type, choices, answer, non-empty fields | code       | yes |
| 1     | Grounding: excerpt must be near-exact copy of the parent text (`difflib.SequenceMatcher` ≥ `quiz_grounding_ratio`, default 0.85) | code       | yes |
| 2     | Answer-key verification: LLM-only given the parent excerpt and question+answer | `_grade_short_llm` | no (LLM) |
| 3     | Dedupe: stemmed-token Jaccard vs all previously accepted questions (`quiz_dedupe_jaccard`, default 0.75) | code      | yes |

Each unit loops up to `quiz_max_batches` (default 12); failed questions are dropped and topped up until the allocation is met. Questions are appended **incrementally** (crash-safe / resumable) to the quiz file after each batch.

**Structured output probe**: One-time test (`_probe()`) on the configured endpoint decides whether to ask the model for strict JSON (`response_format: json_schema`) or fall back to the existing markdown-with-delimiters protocol parsed by `study.parse_questions()`. Probe result is cached in-process.

**Output formats**: Generated quizzes render as Markdown with collapsible `<details>` answers via `study.render_markdown()`. They also export to GIFT (Moodle), CSV, JSON, and Anki `.apkg` through `quiz_export.py`.

### Session 4 — Grading & Analytics (`scripts/quiz_grade.py`)

Grades individual answers when the learner plays the quiz in the Study tab player:

- **mcq**: Exact letter match (`A=B=C`). Returns `{correct, expected, feedback, score}`.
- **true_false**: Normalized coercion (`True/False/yes/no/1/0` → bool). Scored 1.0 or 0.0.
- **fill_blank**: Token-normalized string match (`norm_tokens()`). Accepts exact or containing match.
- **short_answer**: Two-tier approach —
  - Fast path: token-overlap fallback if `quiz_grade_llm=false` or client unavailable.
  - LLM path: small, grounded completion (only the question's own provenance excerpt is sent, no context overload). Returns structured verdict `{score 0..1, feedback}` via json_schema. Degrades gracefully to token overlap on failure.

After finishing an attempt, `compute_analytics()` aggregates accuracy per unit/difficulty/type, flags weak units (accuracy < `quiz_weak_threshold`, default 0.6), and emits **provenance-linked study points** listing exactly which sections the learner struggled with.

Analytics are stored inside the quiz file under `quiz["attempts"]` so interrupted sessions survive restarts. The Web UI shows them in the analytics panel.

### Session 5 — Review & FSRS Spaced Repetition (`scripts/quiz_store.py`)

Missed or weak questions auto-enroll into a shared **review store** (`quizzes/review.json`):

- Each card is keyed `<quiz_id>:<qid>`, carries a full snapshot of the question (with answer), and wraps an `fsrs.Card` object.
- Correct-on-first-attempt starts at a stronger state so first review isn't due immediately; missed questions start un-reviewed and come due right away.
- Users rate cards **again/hard/good/easy**; the scheduler recomputes due-date + stability/difficulty on each rating. Rating history enables a **strength-over-time** chart.
- The Review pane shows due-count, retention-rate, new-cards, and daily-stability buckets — all sourced from the single `review.json` file.

### Session 6 — Course View & Conversational Negotiation (`scripts/server.py` + `web/index.html`)

**Course pane**: Browses a work's syllabus with word counts and a `quizzed` flag drawn from stored quizzes. Also offers per-unit summaries via LLM map-reduce over that unit's sections.

**Conversational quiz negotiation**: Large quiz requests (or those mentioning plan/build/save) route through the planner instead of generating inline. The response is a spec proposal with numbered units the user can edit conversationally:

```
User: make me a hard quiz on machine learning
Agent: ### 🧠 Quiz plan — machine learning  (set: veracrypt1)

Here's a proposal for a **30-question** quiz (topic · balanced).

1. NLP Fundamentals — 8 question(s)
2. Neural Network Architectures — 12 question(s)
...

Reply to adjust, e.g. **"drop unit 1"**, **"double unit 2"**,
**"make it 45 questions"**, **"harder"**, or **"build it"** to generate.
```

Follow-up messages apply deltas (`exclude`, `count`, `difficulty`, `depth`, `types`, per-unit `counts`) and re-run budget allocation. The final `"build it"` triggers SESSION 3's audited builder.

## File layout

Under `quizzes/` (gitignored, never committed):

```
quizzes/
  <id>.json                 — generated quiz (SESSION 3+, shape defined there)
  specs/
    <id>.json               — editable quiz specs from the planner (SESSION 2)
  syllabus/
    <set>/
      <slug>.<parents_hash>.json  — cached LLM outlines, invalidated on re-ingest
  out_of_scope.json         — ledger of refused topic plans
  review.json               — shared FSRS review store (SESSION 5)
```

All writes are **atomic** (tmp file + `os.replace`) so a crash never leaves truncated data. IDs are 12-hex like conversations, using `uuid.uuid4().hex[:12]`.

## Config knobs (tunable via Setup tab or config.local.json)

| Key                                | Default  | Description                                                      |
|------------------------------------|----------|------------------------------------------------------------------|
| `quiz_material_chunks`             | 10       | Max chunks of retrieved material sent per prompt                 |
| `quiz_material_words`              | 3000     | Word cap per chunk                                               |
| `quiz_batch_parents`               | 4        | Parents sent per generation/verification batch                  |
| `quiz_parent_word_cap`             | 500      | Per-parent word limit when building generation material         |
| `quiz_max_batches`                 | 12       | Max parent-batches per unit before giving up                    |
| `quiz_verify_pass`                 | true     | Run LLM answer-key verification (stage 2)                       |
| `quiz_grounding_ratio`             | 0.85     | Min difflib ratio for excerpt grounding (stage 1)               |
| `quiz_dedupe_jaccard`              | 0.75     | Min stemmed-Jaccard similarity to flag duplicate                |
| `llm_structured_output`            | auto     | Probe-enabled JSON generation (auto/off/on)                     |
| `quiz_default_count`               | 10       | Default question count when not specified                        |
| `quiz_output_format`               | markdown | Default export format (`markdown`, `gift`, `csv`, `json`, `apkg`)|
| `quiz_grade_llm`                   | true     | Enable LLM-assisted grading for short-answer questions          |
| `quiz_sample_children`             | 50       | Chunks sampled when falling back to child-sampling (non-parent_child indexes) |
| `quiz_topic_section_pool`          | 120      | Section pool capped when cosine-ranking a topic plan           |
| `quiz_topic_max_works`             | 3        | Max works a topic plan ranks units across                       |
| `quiz_default_depth`               | balanced | Planner depth knob: surface/balanced/deep                       |

When `chunking_strategy != "parent_child"`, `build_quiz` falls back to `_sample_children` which paginates Chroma directly to gather evenly-spaced chunks (controlled by `quiz_sample_children`), bypassing the top-k semantic limit.

## Using the MCP tool

The MCP server exposes the same quiz capabilities as the web UI:

```
mcp> make_quiz(topic="like switch", count=10)
mcp> plan_quiz(request={"mode":"work","work":"The Like Switch","count":30})
mcp> build_quiz(spec_id="<spec-id>")
mcp> get_quiz(quiz_id="<quiz-id>")
mcp> grade_answer(quiz_id="<id>", qid="<qid>", response="C")
mcp> review_queue(set_name="veracrypt1", limit=20)
```

See `scripts/mcp_server.py` for full tool definitions and parameter lists. The MCP tools read the same config keys and share the persistence layer — everything built from one session surfaces in the other.
