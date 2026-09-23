#!/usr/bin/env python3
"""quiz_build.py - SESSION 3 work-specific quiz generation pipeline.

Deterministic, per-work generation: material comes from an ordered walk of the
``parents`` table (never from semantic search) so a quiz covers a work's actual
sections. Each batch of parents is sent to the LLM; every parsed question must
pass the 4-stage audit (scripts/quiz_audit.py) before it is kept; failed
questions are dropped with reasons and the builder tops up until the per-unit
allocation is met or ``quiz_max_batches`` is exhausted. Questions are appended
to the quiz file incrementally (crash-safe/resumable), each carrying a ``qid``
and unit/section tags. Every quiz stores its full build report so accuracy is
measurable per quiz.

Structured output: a one-time capability probe on the configured endpoint
(``llm_structured_output`` = auto|on|off) decides whether to ask the model for
a strict JSON object (response_format json_schema) or fall back to the existing
markdown-with-delimiters protocol (parsed by study.py). The probe result is
cached in-memory for the process lifetime, falling back to offline on failure.

The material format reuses the study-compatible ``--- Source N: <title> ---``
headers so that parsing/provenance logic stays consistent; the parent_id is
carried in a header marker so the audit can map a question back to its real
parent text.
"""

import json
import re

import quiz_audit
import quiz_store

# ─── Structured-output capability probe ──────────────────────────────────────

_STRUCTURED_CACHE: dict = {}


def _structured_mode(cfg) -> str:
    return (cfg.get("llm_structured_output") or "auto").strip().lower()


def structured_output_supported(client, cfg) -> bool:
    """Return True if we should use response_format json_schema for generation.

    - mode "on"  -> True without probing
    - mode "off" -> False
    - mode "auto"-> probe once (cached), fall back to False on any failure
    """
    mode = _structured_mode(cfg)
    if mode == "on":
        return True
    if mode == "off":
        return False
    cached = _STRUCTURED_CACHE.get("auto")
    if cached is not None:
        return cached
    support = _probe(client, cfg)
    _STRUCTURED_CACHE["auto"] = support
    return support


def _probe(client, cfg) -> bool:
    """A minimal completion asking for a JSON object under a json_schema.

    Returns True if the endpoint accepts response_format and returns the
    (eventually parsed) JSON; False on any error.
    """
    schema = {
        "name": "probe",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    try:
        resp = client.chat.completions.create(
            model=cfg.get("llm_model", "default"),
            messages=[{"role": "user",
                       "content": "Reply with {\"ok\": true} exactly."}],
            temperature=0,
            max_tokens=64,
            response_format={"type": "json_schema", "json_schema": schema},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content:
            json.loads(content)  # only care that it parsed
            return True
        return False
    except Exception:
        return False


# ─── Material helpers ────────────────────────────────────────────────────────

def _trim_words(text: str, cap: int) -> str:
    if cap <= 0:
        return text or ""
    words = (text or "").split()
    if len(words) <= cap:
        return text or ""
    return " ".join(words[:cap]) + " …[truncated]"


def build_parent_material(parents: list, word_cap: int) -> str:
    """Build a ``--- Source N: <title> (parent=<id>) [KIND] ---`` material block.

    ``parents`` is a list of dicts with keys {parent_id, title, section_title,
    text}. Returns a string suitable for the generation prompt.
    """
    parts = []
    for i, p in enumerate(parents, 1):
        title = p.get("title") or "Unknown"
        body = _trim_words(p.get("text") or "", word_cap)
        header = (f"--- Source {i}: {title} (parent={p.get('parent_id') or ''}) "
                  f"[SECTION: {p.get('section_title') or ''}] ---")
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


# ─── Model I/O ───────────────────────────────────────────────────────────────

_STUDY_TYPES = ("mcq", "short_answer", "true_false", "fill_blank")


def _gen_prompt(title_label, unit_title, count, material, types, difficulty,
                structured: bool) -> str:
    """Build the generation prompt.

    In structured mode we ask for a JSON object with a ``questions`` array; in
    delimiter mode we use the study.py markdown-with-delimiters format so the
    existing parser handles it. In both cases the model MUST cite the PARENT id
    so the audit can ground the excerpt.

    The EXCERPT instructions are strongly worded because small models (e.g. 1.5B)
    tend to paraphrase rather than quote — but the grounding audit requires near-
    exact string matches via difflib.SequenceMatcher ≥ 0.85. Verbatim quoting is
    enforced by both the prompt text and a separate system-level note.
    """
    types_line = ", ".join(types) if types else "mcq, short_answer, true_false, fill_blank"
    base = (
        f"You are writing practice questions about \"{unit_title}\" "
        f"(from the work \"{title_label}\"). Base EVERY question ONLY on the "
        f"source material below. Produce exactly {count} questions.\n"
        f"Allowed types: {types_line}. Difficulty mix: {difficulty}.\n"
        "Each question MUST cite, from the source material, the PARENT id it "
        "draws from (the id in the header like `(parent=xxx)`).\n\n"
        "EXCERPT RULES (critical — your answers will be audited):\n"
        "- Each question must include ONE sentence of evidence copied EXACTLY "
        "and VERBATIM from the source material.\n"
        "- Copy the sentence character-for-character. Do NOT paraphrase, rewrite, "
        "summarize, or normalize any words.\n"
        "- If you cannot find an exact matching sentence, omit the question rather "
        "than inventing one.\n\n"
        "--- Source material ---\n\n" + (material or "(no material provided)"))
    if structured:
        return base + (
            "\n\nReturn ONLY a JSON object of the form:\n"
            '{"questions": [{"q": "...", "type": "mcq|short_answer|true_false|'
            'fill_blank", "difficulty": "easy|medium|hard", "choices": '
            '["A) ...", "B) ...", ...] (empty array for non-mcq), "answer": '
            '"<answer letter for mcq, else the correct text>", "provenance": '
            '{"title": "<book title>", "section_title": "...", "parent_id": '
            '"<parent id>", "excerpt_snippet": "<VERBATIM exact-copy sentence>"}}]}')
    base += (
        "\n\nFor each question output a block separated by a line of 5 dashes "
        "('----------'), with these exact fields:\n"
        "QUESTION: <the question>\n"
        "TYPE: <type>\n"
        "DIFFICULTY: easy | medium | hard\n"
        "A) <choice 1>\nB) <choice 2>\nC) <choice 3>\nD) <choice 4>\n"
        "(omit A-D for short_answer/fill_blank; true_false has A) true / B) false)\n"
        "ANSWER: <correct letter for mcq/true_false, else the correct text>\n"
        "SOURCE: <book title>\n"
        "SECTION: <chapter/section title>\n"
        "PARENT: <parent id>\n"
        "EXCERPT: <VERBATIM exact-copy sentence from source>")
    return base


def _call_llm(client, cfg, prompt, structured: bool):
    """Call the LLM with both a system message (output rules) and user message
    (the actual quiz generation prompt). A system role helps Qwen2.5 small models
    follow the verbatim-quoting rule more reliably than a bare user message."""
    system = (
        "You are an expert educator writing practice questions for students. "
        "When asked to produce an EXCERPT or QUOTE, you MUST copy text exactly "
        "and verbatim from the provided source material — do NOT paraphrase, "
        "rewrite, summarize, or normalize any words. Every answer must be "
        "grounded in the source."
    )
    kwargs = dict(
        model=cfg.get("llm_model", "default"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=int(cfg.get("max_tokens_quiz", 4096) or 4096),
    )
    if structured:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "questions",
                "schema": _questions_schema(),
                "strict": True,
            },
        }
    return client.chat.completions.create(**kwargs)


def _questions_schema() -> dict:
    choice = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "type": {"type": "string",
                                 "enum": list(_STUDY_TYPES)},
                        "difficulty": {"type": "string",
                                       "enum": ["easy", "medium", "hard"]},
                        "choices": {
                            "type": "array",
                            "items": choice,
                        },
                        "answer": {"type": "string"},
                        "provenance": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "section_title": {"type": "string"},
                                "parent_id": {"type": "string"},
                                "excerpt_snippet": {"type": "string"},
                            },
                            "required": ["title", "section_title", "parent_id",
                                         "excerpt_snippet"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["q", "type", "difficulty", "choices", "answer",
                                 "provenance"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def _flat_choices(choices) -> list:
    """Convert a list of 'A) ...' strings (or raw strings without letters)
    from structured output into the canonical IR 'A) ...' list."""
    out = []
    letters = "ABCDEFGH"
    for i, c in enumerate(choices or []):
        c = (c or "").strip()
        if not c:
            continue
        if re.match(r"^[A-D]\s*[).:]", c):
            out.append(c)
        elif i < len(letters):
            out.append(f"{letters[i]}) {c}")
        else:
            out.append(c)
    return out


def _normalize_json_question(item: dict) -> dict:
    """Normalize a raw structured-output question item into the study IR."""
    if not isinstance(item, dict):
        return None
    qtype = str(item.get("type") or "mcq").strip().lower()
    if qtype not in _STUDY_TYPES:
        qtype = "mcq"
    diff = str(item.get("difficulty") or "medium").strip().lower()
    if diff not in ("easy", "medium", "hard"):
        diff = "medium"
    choices = _flat_choices(item.get("choices") or []) if qtype == "mcq" else None
    prov = item.get("provenance") or {}
    return {
        "q": str(item.get("q") or "").strip(),
        "choices": choices,
        "answer": str(item.get("answer") or "").strip(),
        "difficulty": diff,
        "type": qtype,
        "provenance": {
            "title": str(prov.get("title") or "").strip(),
            "section_title": str(prov.get("section_title") or "").strip(),
            "parent_id": str(prov.get("parent_id") or "").strip(),
            "excerpt_snippet": str(prov.get("excerpt_snippet") or "").strip()[:400],
        },
    }


def generate_batch(material, label, unit_title, count, cfg, client,
                   allowed_types=None, difficulty="mixed", structured=False) -> tuple:
    """Generate a batch of questions from ``material``.

    Returns (questions: list of IR dicts, err: str|None). Reuses study.py's
    parse_questions for the delimiter path and a JSON normalize for the
    structured path. The count is a target — the list may be shorter.
    """
    import study
    types = allowed_types or list(_STUDY_TYPES)
    prompt = _gen_prompt(label, unit_title, count, material, types, difficulty,
                         structured)
    try:
        resp = _call_llm(client, cfg, prompt, structured)
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return [], str(e)
    if not text:
        return [], "empty model output"
    if structured:
        qs = _parse_structured(text)
        return qs, None
    qs = study.parse_questions(text)
    return qs, None


def _parse_structured(text: str) -> list:
    """Parse a JSON object with a ``questions`` array; tolerant of markdown
    fences and leading prose because some servers wrap JSON."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except Exception:
        # Try to find the first { ... } spanning the whole answer.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start:end + 1])
        except Exception:
            return []
    items = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        norm = _normalize_json_question(it)
        if norm and norm.get("q"):
            out.append(norm)
    return out


# ─── Inline LLM answer-key verification (stage 2) ────────────────────────────

def verify_questions(verified: dict, parent_texts: dict, cfg, client) -> dict:
    """LLM-verify a set of {qid: question} sharing one parent text.

    Given ONLY the parent text + question + choices + marked answer, return a
    structured verdict (supported? yes/no + note). Returns {qid: bool} after
    dropping unsupported. Batched per parent to amortize context.
    """
    if not verified:
        return {}
    model = cfg.get("llm_model", "default")
    groups = {}
    for qid, q in verified.items():
        pid = (q.get("provenance") or {}).get("parent_id", "")
        groups.setdefault(pid, []).append((qid, q))

    accepted = {}
    for pid, items in groups.items():
        pt = (parent_texts.get(pid) or {}).get("text", "")
        blocks = []
        for qid, q in items:
            choices = q.get("choices")
            blocks.append(
                f"Q: {q.get('q')}\n"
                f"Choices: {', '.join(choices or [])}\n"
                f"Marked answer: {q.get('answer')}\n")
        prompt = (
            "Verify whether each question's MARKED ANSWER is SUPPORTED by the "
            "passage (base it ONLY on the passage). Respond as a JSON object "
            "{'answers': [{'index': 0, 'supported': true, 'note': '...'}]} "
            "where index matches each question's position. supported=true only "
            "if the passage directly supports the marked answer.\n\n"
            "--- Passage ---\n" + _trim_words(pt, 2000) +
            "\n\n--- Questions ---\n" + "\n".join(blocks))
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "verdicts",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "supported": {"type": "boolean"},
                                        "note": {"type": "string"},
                                    },
                                    "required": ["index", "supported", "note"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["answers"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }},
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            verdicts = {int(v.get("index")): bool(v.get("supported"))
                        for v in data.get("answers", [])}
        except Exception:
            return {qid: False for qid, _ in items}
        for i, (qid, _q) in enumerate(items):
            if verdicts.get(i, False):
                accepted[qid] = True
    return accepted


# ─── The builder ─────────────────────────────────────────────────────────────

def _parent_texts_for(parent_ids, set_name) -> dict:
    """Load parent section texts directly from manifest.db `parents`.

    Mirrors agent._load_parent_texts but avoids importing agent (which pulls in
    chromadb/sentence_transformers) so the builder stays import-light.
    Returns {parent_id: {text, title, section_title}}.
    """
    ids = [p for p in dict.fromkeys(parent_ids or []) if p]
    if not ids:
        return {}
    import sqlite3
    from _paths import rag_root as _rag_root
    db = _rag_root() / "manifest.db"
    if not db.exists():
        return {}
    out = {}
    try:
        conn = sqlite3.connect(str(db))
        for i in range(0, len(ids), 400):
            batch = ids[i:i + 400]
            q = ("SELECT parent_id, text, title, section_title FROM parents "
                 f"WHERE parent_id IN ({','.join('?' * len(batch))})")
            args = list(batch)
            if set_name:
                q += " AND set_name = ?"
                args.append(set_name)
            for pid, text, title, sec_title in conn.execute(q, args):
                out[pid] = {"text": text or "", "title": title or "",
                            "section_title": sec_title or ""}
        conn.close()
    except Exception as e:
        print(f"[quiz_build] parent store lookup failed ({e})", flush=True)
        return {}
    return out


def build_quiz(spec, set_name, cfg, client, stream_cb=None, progress_cb=None,
               collection=None, embedder=None) -> dict:
    """Build a quiz from a persisted QuizSpec, writing the quiz file.

    Returns the final quiz dict ({id, title, set_name, questions, report}).
    ``stream_cb`` receives ``{"type": "unit", ...}`` / ``{"type":
    "accept", ...}`` deltas; ``progress_cb`` receives a human-readable message
    roughly per unit (used to keep SSE alive).
    """
    spec_id = (spec or {}).get("id")
    title = (spec or {}).get("title") or (spec or {}).get("work") or "Quiz"
    units = [u for u in (spec or {}).get("units", []) if u.get("selected", True)]
    units.sort(key=lambda u: u.get("ordinal", 0))

    batch_p = max(1, int(cfg.get("quiz_batch_parents", 4) or 4))
    word_cap = max(50, int(cfg.get("quiz_parent_word_cap", 500) or 500))
    max_batches = max(1, int(cfg.get("quiz_max_batches", 12) or 12))
    ground_ratio = float(cfg.get("quiz_grounding_ratio", 0.85) or 0.85)
    dedupe_jac = float(cfg.get("quiz_dedupe_jaccard", 0.75) or 0.75)
    verify_pass = bool(cfg.get("quiz_verify_pass", True))
    structured = structured_output_supported(client, cfg)

    chunking = (cfg.get("chunking_strategy") or "flat")
    has_parents = chunking == "parent_child"

    # Quiz file created up front so partial progress survives a crash.
    quiz = quiz_store.get_quiz(spec_id) if spec_id else None
    if quiz is None:
        new_quiz = {
            "title": title,
            "set_name": set_name,
            "spec_id": spec_id,
            "status": "building",
            "questions": [],
            "report": None,
        }
        if spec_id:
            new_quiz["id"] = spec_id
        quiz = quiz_store.create_quiz(new_quiz)
    else:
        quiz["status"] = "building"
        quiz.setdefault("questions", [])
        quiz = quiz_store.update_quiz(quiz)

    report = {
        "structured_output": structured,
        "chunking_parents": has_parents,
        "units_requested": len(units),
        "per_unit": {},
        "drop_reasons": {},
        "warnings": [],
        "total_requested": 0,
        "total_accepted": 0,
        "stages": {"parse": 0, "grounding": 0, "verify": 0, "dedupe": 0},
    }
    if not has_parents:
        report["warnings"].append(
            "chunking_strategy is not parent_child; using even-spread child-"
            "chunk sampling fallback.")

    accepted = list(quiz.get("questions") or [])
    qid_counter = len(accepted)

    def progress(msg):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    for unit in units:
        uid = unit.get("unit_id") or f"u{unit.get('ordinal')}"
        alloc = max(0, int(unit.get("allocation", 0) or 0))
        if alloc <= 0:
            continue
        report["total_requested"] += alloc
        unit_parent_ids = [p for p in (unit.get("parent_ids") or []) if p]

        # Load this unit's parent texts (deterministic parents walk).
        parent_texts = _parent_texts_for(unit_parent_ids, set_name) if unit_parent_ids else {}

        report["per_unit"][uid] = {
            "requested": alloc, "accepted": 0, "batches": 0,
            "drops": {},
        }
        progress(f"Generating {alloc} questions for \"{unit.get('title') or uid}\" …")

        # Ordered parents for this unit — fall back to child sampling if absent.
        parents_ordered = []
        for pid in unit_parent_ids:
            p = parent_texts.get(pid)
            if p:
                parents_ordered.append({
                    "parent_id": pid,
                    "title": p.get("title") or unit.get("work") or title,
                    "section_title": p.get("section_title") or unit.get("title") or "",
                    "text": p.get("text") or "",
                })
        if not parents_ordered:
            parents_ordered = _sample_children(unit, set_name, cfg, collection,
                                               embedder, word_cap)

        unit_accepted = 0
        unit_batches = 0
        # Batch the parents in order; if the pool is empty there is nothing to
        # generate from, so this unit is skipped with a warning.
        pool = parents_ordered or []
        if not pool:
            report["per_unit"][uid]["accepted"] = 0
            report["warnings"].append(
                f"Unit {uid}: no parent text available; skipped.")
            continue
        made_progress_this_unit = False
        while unit_accepted < alloc and unit_batches < max_batches:
            unit_batches += 1
            # Contiguous slice through the ordered parents pool (cycling so a
            # small pool can still be re-mined toward the allocation).
            start = ((unit_batches - 1) * batch_p) % len(pool)
            batch = (pool[start:start + batch_p] +
                     pool[:max(0, batch_p - (len(pool) - start))])
            batch = [p for p in batch if p]
            if not batch:
                break
            want = alloc - unit_accepted
            material = build_parent_material(batch, word_cap)
            label = title
            unit_title = unit.get("title") or uid
            qs, err = generate_batch(
                material, label, unit_title, want, cfg, client,
                allowed_types=(spec or {}).get("types"),
                difficulty=(spec or {}).get("difficulty", "mixed"),
                structured=structured)
            if err:
                report["drop_reasons"]["llm_error"] = \
                    report["drop_reasons"].get("llm_error", 0) + 1
                report["per_unit"][uid]["drops"]["llm_error"] = \
                    report["per_unit"][uid]["drops"].get("llm_error", 0) + 1
                if not made_progress_this_unit and unit_batches >= 3:
                    report["warnings"].append(
                        f"Unit {uid}: LLM failed {unit_batches} batches ({err}); "
                        "stopping this unit.")
                    break
                continue

            if not qs:
                report["drop_reasons"]["empty_batch"] = \
                    report["drop_reasons"].get("empty_batch", 0) + 1
                if not made_progress_this_unit and unit_batches >= 4:
                    report["warnings"].append(
                        f"Unit {uid}: {unit_batches} empty batches; stopping.")
                    break
                continue

            # Resolve each question's parent for the deterministic audit.
            by_pid = {p["parent_id"]: p for p in parents_ordered}
            staged = []
            for q in qs:
                prov = q.get("provenance") or {}
                pid = prov.get("parent_id", "")
                parent = by_pid.get(pid) if pid else None
                ok, reason = quiz_audit.run_deterministic_audit(q, parent,
                                                                ground_ratio)
                if not ok:
                    report["stages"]["parse"] += 1
                    report["drop_reasons"][reason] = \
                        report["drop_reasons"].get(reason, 0) + 1
                    report["per_unit"][uid]["drops"][reason] = \
                        report["per_unit"][uid]["drops"].get(reason, 0) + 1
                    continue
                staged.append(q)

            # Stage 2: inline LLM answer-key verification (batched per parent).
            if verify_pass and staged:
                # keyed by index so verify returns accepted membership
                verify_input = {str(i): q for i, q in enumerate(staged)}
                accepted_idx = verify_questions(verify_input, parent_texts,
                                                cfg, client)
                kept, dropped = [], 0
                for i, q in enumerate(staged):
                    if str(i) in accepted_idx:
                        kept.append(q)
                    else:
                        dropped += 1
                        report["stages"]["verify"] += 1
                if dropped:
                    report["drop_reasons"]["verify_failed"] = \
                        report["drop_reasons"].get("verify_failed", 0) + 1
                    report["per_unit"][uid]["drops"]["verify_failed"] = \
                        report["per_unit"][uid]["drops"].get("verify_failed", 0) + 1
                staged = kept

            # Stage 3: cross-batch dedupe (stemmed Jaccard vs all accepted).
            kept = []
            for q in staged:
                if quiz_audit.is_duplicate(q, accepted, dedupe_jac):
                    report["stages"]["dedupe"] += 1
                    report["drop_reasons"]["duplicate"] = \
                        report["drop_reasons"].get("duplicate", 0) + 1
                    report["per_unit"][uid]["drops"]["duplicate"] = \
                        report["per_unit"][uid]["drops"].get("duplicate", 0) + 1
                    continue
                kept.append(q)
            staged = kept

            # Tag + adopt the survivors.
            for q in staged:
                qid_counter += 1
                now_uid = uid
                q["qid"] = f"{quiz.get('id', 'quiz')}-{qid_counter:04d}"
                q["unit"] = now_uid
                q["section_title"] = (q.get("provenance") or {}).get(
                    "section_title") or unit.get("title") or ""
            accepted.extend(staged)
            unit_accepted += len(staged)
            made_progress_this_unit = True
            # Crash-safe incremental write after each real batch.
            quiz["questions"] = accepted
            quiz = quiz_store.update_quiz(quiz)
            if stream_cb:
                try:
                    stream_cb({"type": "unit", "unit": uid,
                               "accepted": unit_accepted,
                               "requested": alloc})
                except Exception:
                    pass

        report["per_unit"][uid]["accepted"] = unit_accepted
        report["per_unit"][uid]["batches"] = unit_batches
        if unit_accepted < alloc:
            report["warnings"].append(
                f"Unit {uid}: only {unit_accepted}/{alloc} questions after "
                f"{unit_batches} batch(es).")

    report["total_accepted"] = len(accepted)
    quiz["questions"] = accepted
    quiz["status"] = "ready"
    quiz["report"] = report
    quiz = quiz_store.update_quiz(quiz)
    if stream_cb:
        try:
            stream_cb({"type": "done", "question_count": len(accepted),
                       "id": quiz.get("id")})
        except Exception:
            pass
    return quiz


def _sample_children(unit, set_name, cfg, collection, embedder, word_cap) -> list:
    """Even-spread child-chunk sampling fallback for non-parent_child indexes.

    Returns a list of pseudo-parent dicts (as much as could be sampled) or []
    when no collection/embedder is available. This is a best-effort fallback so
    generation never hard-fails on an index without parents.
    """
    try:
        import agent as _agent
        if collection is None or embedder is None:
            return []
        work = unit.get("work") or ""
        if not work:
            return []
        hits = _agent.retrieve(
            unit.get("title") or work, embedder, collection,
            top_k=min(40, int(cfg.get("retrieval_top_k", 10) or 10) * 4),
            where_extra={"title": {"$in": [work]}} if work else None)
        if not hits:
            return []
        n = len(hits)
        stride = max(1, n // 6)
        sampled = []
        for h in hits[::stride][:6]:
            meta = h.get("metadata") or {}
            text = h.get("document") or ""
            if not text:
                continue
            sampled.append({
                "parent_id": h.get("id") or meta.get("parent_id") or "",
                "title": meta.get("title") or work,
                "section_title": meta.get("section_title") or "",
                "text": text,
            })
        return sampled
    except Exception as e:
        print(f"[quiz_build] child sampling failed ({e})", flush=True)
        return []
