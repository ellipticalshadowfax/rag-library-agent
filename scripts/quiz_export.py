#!/usr/bin/env python3
"""quiz_export.py - Renderers that turn a stored quiz's Question[] into
portable formats.

SESSION 6 wires the previously-dead `quiz_output_format` config key to a
default format and exposes these renderers through
`GET /api/quiz/<id>/export?format=...`. study.py was designed so new export
formats only add renderers and never touch generation/validation — this module
is the "add a renderer" home.

Formats:
  markdown  (default)  - study.render_markdown (human-readable, collapsible)
  gift      - Moodle GIFT text (MCQ ==GIFT== format, TF, short answer)
  csv       - flat rows per question: q,type,difficulty,choices,answer,source
  json      - the raw Question[] (plus quiz id/title), pretty-printed
  apkg      - Anki deck (.apkg) via genanki (optional dep, try/except import)

All renderers are pure (take Question[] + optional meta, return str/bytes).
Heavy imports (genanki) are deferred inside the function that needs them.
"""

import io
import json

import study


# ─── Format registry ─────────────────────────────────────────────────────────

FORMATS = ("markdown", "gift", "csv", "json", "apkg")


def _choice_parts(choices):
    """Return (letter, text) pairs for a question's 'A) ...' choice list."""
    import re
    out = []
    for c in choices or []:
        m = re.match(r"^\s*([A-Za-z])\s*[).:]\s*(.*)$", c)
        if m:
            out.append((m.group(1).upper(), m.group(2).strip()))
        else:
            out.append((None, c.strip()))
    return out


# ─── Renderers ───────────────────────────────────────────────────────────────

def render_markdown(quiz):
    """Human-readable Markdown (the default / web format)."""
    import study as _s
    return _s.render_markdown(quiz.get("questions") or [])


def render_gift(quiz):
    """Moodle GIFT format. https://docs.moodle.org/GIFT_format"""
    lines = ["// Quiz: %s" % (quiz.get("title") or "untitled")]
    for i, qu in enumerate(quiz.get("questions") or [], 1):
        qtype = (qu.get("type") or "mcq").strip().lower()
        qtext = _gift_escape(qu.get("q") or "")
        answer = (qu.get("answer") or "").strip()
        if qtype in ("mcq",):
            parts = _choice_parts(qu.get("choices"))
            # GIFT marks the correct line with '=' and distractors with '~'.
            # Our answer key is the correct letter (e.g. "A").
            key_letter = "".join(ch for ch in answer if ch.isalpha()).upper()
            items = []
            for letter, text in parts:
                marker = "=" if (letter and key_letter and letter == key_letter) else "~"
                items.append("%s%s" % (marker, _gift_escape(text)))
            lines.append("%s {\n  %s\n}" % (qtext, "\n  ".join(items)))
        elif qtype == "true_false":
            tf = "TRUE" if str(answer).strip().lower() in ("true", "t") else "FALSE"
            lines.append("%s{%s}" % (qtext, tf))
        elif qtype == "fill_blank":
            lines.append("%s{=%s}" % (_gift_escape((qu.get("q") or "")), _gift_escape(answer)))
        else:  # short_answer
            lines.append("%s{=%s}" % (_gift_escape((qu.get("q") or "")), _gift_escape(answer)))
        lines.append("")
    return "\n".join(lines)


def _gift_escape(s):
    """Escape GIFT-special characters in question/answer text."""
    return (s or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def render_csv(quiz):
    """Flat rows per question."""
    import csv as _csv
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["q", "type", "difficulty", "choices", "answer", "source",
                "section", "unit", "qid"])
    for qu in quiz.get("questions") or []:
        prov = qu.get("provenance") or {}
        w.writerow([
            qu.get("q") or "",
            qu.get("type") or "mcq",
            qu.get("difficulty") or "",
            "; ".join(qu.get("choices") or []),
            qu.get("answer") or "",
            prov.get("title") or "",
            prov.get("section_title") or "",
            qu.get("unit") or "",
            qu.get("qid") or "",
        ])
    return buf.getvalue()


def render_json(quiz):
    """Raw Question[] plus quiz identity, pretty-printed."""
    qids = quiz.get("questions") or []
    return json.dumps(_json_payload(quiz), indent=2, ensure_ascii=False)


def _json_payload(quiz):
    return {
        "quiz_id": quiz.get("id"),
        "title": quiz.get("title"),
        "set_name": quiz.get("set_name"),
        "question_count": len(quiz.get("questions") or []),
        "questions": quiz.get("questions") or [],
    }


def render_apkg(quiz, deck_name=None):
    """Anki .apkg bytes via genanki (optional dependency)."""
    try:
        import genanki
    except Exception as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "genanki is not installed; install 'genanki' to export .apkg files."
        ) from e

    model = genanki.Model(
        1607392319,
        "RAG Quiz (Basic)",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Front}}",
            "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
        }],
    )
    deck = genanki.Deck(2059400110, deck_name or (quiz.get("title") or "RAG Quiz"))
    for qu in quiz.get("questions") or []:
        front = qu.get("q") or ""
        choices = qu.get("choices") or []
        if choices:
            front += "\n" + "\n".join(choices)
        back = "**%s**" % (qu.get("answer") or "")
        prov = qu.get("provenance") or {}
        src = prov.get("title") or ""
        if prov.get("section_title"):
            src += " — %s" % prov["section_title"]
        if src:
            back += "\n\nSource: %s" % src
        if prov.get("excerpt_snippet"):
            back += "\n\n%s" % prov["excerpt_snippet"]
        deck.add_note(genanki.Note(model=model, fields=[front, back]))
    pkg = genanki.Package(deck)
    buf = io.BytesIO()
    pkg.write_to_file(buf)
    return buf.getvalue()


# ─── Dispatch ────────────────────────────────────────────────────────────────

def export_quiz(quiz: dict, fmt: str = "markdown"):
    """Return (payload, mime, is_binary) for a quiz dict and format name.

    ``fmt`` falls back to "markdown" on unknown values. ``is_binary`` is True
    only for .apkg so the caller can send raw bytes with the right mime.
    """
    fmt = (fmt or "markdown").strip().lower()
    if fmt == "gift":
        return render_gift(quiz), "text/plain; charset=utf-8", False
    if fmt == "csv":
        return render_csv(quiz), "text/csv; charset=utf-8", False
    if fmt == "json":
        return render_json(quiz), "application/json; charset=utf-8", False
    if fmt == "apkg":
        return render_apkg(quiz), "application/octet-stream", True
    # markdown + unknown → markdown
    return render_markdown(quiz), "text/markdown; charset=utf-8", False
