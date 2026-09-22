#!/usr/bin/env python3
"""study.py - Structured quiz/test generation for the RAG library.

Questions are generated as STRUCTURED objects (the intermediate
representation) and rendered separately. This keeps future export formats
(GIFT, PDF, CSV, HTML) cheap: add a renderer, don't refactor generation.

A Question is a dict:
    {
        "q": "Question text here.",
        "choices": ["A) ...", "B) ..."],          // null for short-answer
        "answer": "Correct answer text.",
        "difficulty": "easy|medium|hard",
        "type": "mcq|short_answer|true_false|fill_blank",
        "provenance": {                            // citation for trust
            "title", "section_title", "parent_id", "excerpt_snippet",
        },
    }

Generation flow:
  1. Material selection (the caller passes the source excerpts/titles).
  2. Prompt the LLM with a strict markdown-with-delimiters format that small
     local models can reliably produce, then parse it back into Question[].
  3. Validate provenance on every question.
  4. Render via render_markdown(questions) with collapsible <details> answers.

Heavy imports are deferred.
"""

import json
import re


_SOURCE_HEADER_RE = re.compile(
    r"(?m)^---\s*Source\s+\d+:\s*(.*?)\s*(?:\(.*?\))?\s*(?:\[(?:FICTION|NON-FICTION)\])?"
    r"(?:,\s*p\.\d+)?\s*---")


def _clean(s) -> str:
    return (s or "").strip()


def source_titles(material: str) -> list:
    """Extract the source-book titles cited in the provided quiz material.

    The caller builds ``material`` via ``agent.build_context``, which emits one
    ``--- Source N: <title> (...) [KIND], p.X ---`` header per retrieved chunk.
    Those are the *only* books a model is allowed to attribute a question to, so
    they form the allowlist for provenance validation.
    """
    titles = []
    for m in _SOURCE_HEADER_RE.finditer(material or ""):
        t = m.group(1).strip()
        if t and t.lower() != "unknown" and t not in titles:
            titles.append(t)
    return titles


def _norm_title(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def title_matches(cite: str, valid_titles: list) -> bool:
    """Fuzzy-match a model-cited source title against the material's allowlist.

    Exact match, containment (either direction — small models often truncate or
    paraphrase a long title), or >=2 shared significant tokens all count as a
    match. A title unrelated to the material (e.g. a hallucinated work) shares
    no tokens and is rejected.
    """
    c = _norm_title(cite)
    if not c:
        return False
    cwords = {w for w in c.split() if len(w) > 1}
    for t in valid_titles:
        tn = _norm_title(t)
        if not tn:
            continue
        if tn == c or (tn and (tn in c or c in tn)):
            return True
        twords = {w for w in tn.split() if len(w) > 1}
        if len(twords & cwords) >= 2:
            return True
    return False


def _parse_blocks(text: str):
    """Split text on a delimiter line of 5+ dashes and trim the parts."""
    parts = re.split(r"(?m)^-{5,}\s*$", text or "")
    return [p.strip() for p in parts if p.strip()]


def parse_questions(text: str) -> list:
    """Parse the model's markdown-with-delimiters output into Question[].

    Supported input format (one block per question, separated by a line of
    5+ dashes), e.g.:

        QUESTION: Who is the author of Black Beauty?
        TYPE: mcq
        DIFFICULTY: medium
        A) Anna Sewell
        B) Charles Dickens
        ANSWER: A
        SOURCE: Black Beauty
        SECTION: Ch. 1
        EXCERPT: Anna Sewell wrote Black Beauty in 1877.
        ----------

    For short-answer questions the ANSWER line holds the free-text answer and
    there are no A/B/C choices.
    """
    blocks = _parse_blocks(text)
    out = []
    for b in blocks:
        q = _field(b, "QUESTION")
        if not q:
            continue
        qtype = (_field(b, "TYPE") or "mcq").strip().lower()
        if qtype not in ("mcq", "short_answer", "true_false", "fill_blank"):
            qtype = "mcq"
        diff = (_field(b, "DIFFICULTY") or "medium").strip().lower()
        if diff not in ("easy", "medium", "hard"):
            diff = "medium"
        choices = _choices(b)
        answer = _field(b, "ANSWER") or ""
        prov = {
            "title": _field(b, "SOURCE") or "",
            "section_title": _field(b, "SECTION") or "",
            "parent_id": _field(b, "PARENT") or "",
            "excerpt_snippet": (_field(b, "EXCERPT") or "")[:400],
        }
        out.append({
            "q": q,
            "choices": choices,
            "answer": answer,
            "difficulty": diff,
            "type": qtype,
            "provenance": prov,
        })
    return out


def _field(block: str, key: str):
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*:\s*(.*)$", block)
    if not m:
        return ""
    return m.group(1).strip()


def _choices(block: str):
    out = []
    for m in re.finditer(r"(?mi)^\s*([A-D])\s*[).]\s*(.+)$", block):
        letter, text = m.group(1), m.group(2).strip()
        if not text:
            continue
        if letter.upper() == "E":
            break
        out.append(f"{letter}) {text}")
    return out or None


def generate_quiz(topic_or_title, material, cfg, client):
    """Generate quiz questions from retrieved ``material``.

    ``material`` is a string of source excerpts (titles + text) to quiz on.
    Returns a list of Question dicts (may be empty on parse failure).
    """
    count = int(cfg.get("quiz_default_count", 10) or 10)
    model = cfg.get("llm_model", "default")
    prompt = (
        f"Create a practice quiz on the topic \"{topic_or_title}\" based ONLY "
        f"on the source material below. Produce exactly {count} questions.\n\n"
        "For each question output a block separated by a line of 5 dashes "
        "('----------'), with these exact fields:\n"
        "QUESTION: <the question>\n"
        "TYPE: mcq | short_answer | true_false | fill_blank\n"
        "DIFFICULTY: easy | medium | hard\n"
        "A) <choice 1>\nB) <choice 2>\nC) <choice 3>\nD) <choice 4>\n"
        "(omit A-D lines for short_answer/fill_blank)\n"
        "ANSWER: <correct letter, e.g. A, or the correct text>\n"
        "SOURCE: <title of the book this question draws from>\n"
        "SECTION: <chapter or section title if known>\n"
        "EXCERPT: <one sentence of evidence quoted from the material>\n"
        "----------\n"
        "Test understanding, not just recall. Every question MUST cite its "
        "source book and quote evidence.\n\n"
        "--- Source material ---\n\n" + (material or "(no material provided)"))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.get("llm_temperature", 0.3),
            max_tokens=int(cfg.get("max_tokens_quiz", 4096) or 4096),
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return [], str(e)

    questions = parse_questions(text)
    # Drop questions that cite a source book that is not actually in the
    # material. Small local models sometimes hallucinate a plausible-sounding
    # but unrelated book (e.g. a horsemanship quiz citing "Vagus Nerve and
    # Polyvagal Theory"); validating provenance against the material's source
    # titles keeps every question grounded and citable.
    valid = source_titles(material)
    if valid:
        questions = [q for q in questions
                      if title_matches((q.get("provenance") or {}).get("title") or "", valid)]
    return questions, None


def render_markdown(questions: list) -> str:
    """Render Question[] as clean Markdown with collapsible <details> answers."""
    if not questions:
        return "_No questions could be generated._"
    lines = []
    for i, qu in enumerate(questions, 1):
        t = qu.get("type", "mcq")
        diff = qu.get("difficulty", "medium")
        lines.append(f"**{i}. {qu.get('q','')}**")
        lines.append(f"<span class=\"small muted\">*{t} · {diff}*</span>")
        choices = qu.get("choices")
        if choices:
            for c in choices:
                lines.append(f"- {c}")
        lines.append("")
        answer_block = [f"**Answer:** {qu.get('answer','')}"]
        prov = qu.get("provenance") or {}
        if prov.get("title") or prov.get("section_title"):
            src = prov.get("title", "?")
            if prov.get("section_title"):
                src += f" — {prov['section_title']}"
            answer_block.append(f"*Source: {src}*")
        if prov.get("excerpt_snippet"):
            answer_block.append(f"> {prov['excerpt_snippet']}")
        inner = "<br>".join(answer_block)
        lines.append(f"<details><summary>Show answer</summary>\n\n{inner}\n\n</details>")
        lines.append("")
    return "\n".join(lines)
