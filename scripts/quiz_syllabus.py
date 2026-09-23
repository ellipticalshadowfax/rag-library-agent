#!/usr/bin/env python3
"""quiz_syllabus.py - deterministic section inventory + one cached LLM outline
pass per work that turns the inventory into a hierarchical syllabus.

The "syllabus" is the work's internal structure used later by the planner
(scope resolution, depth knob, budget allocation) and by the SESSION 6 course
view. It does NOT contain question material; it just labels the units.

A unit (dict) has:
    {
        "unit_id": str,          # stable id, e.g. "u0", "u1", ...
        "title": str,            # section_title from parents (fallback ordinal)
        "ordinal": int,          # from parents
        "parent_ids": [str],     # the parents-table row(s) backing this unit
        "subtopics": [str],      # LLM-assigned labels within the unit
        "words": int,            # total words (for budget allocation)
    }

The syllabus is cached to quizzes/syllabus/<set>/<slug>.<parents_hash>.json so a
single LLM call is made per distinct parents-table state. Re-ingest that changes
ordinal / parent_id / section text length invalidates the cache automatically.
"""

import hashlib
import re

import quiz_store


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return s or "work"


def inventory(title: str, set_name: str) -> list:
    """Read-only ordered section inventory of a work (see quiz_store)."""
    return quiz_store.inventory_sections(title, set_name)


def _build_syllabus_prompt(title: str, sections: list) -> str:
    """Put the section list + first ~120 words of each section into a prompt."""
    parts = [f"Below are the sections of the work \"{title}\". For EACH section "
             f"given, output one line in EXACTLY this format, with no other text "
             "or markdown:\n\n"
             "UNIT <ordinal>: <short title> | SUBTOPICS: <comma-separated "
             "subtopic labels>\n\n"
             "The subtopics should be 2-5 short labels describing what that "
             "section covers. Base everything ONLY on the section text "
             "provided.\n\n--- Sections ---"]
    for s in sections:
        head = " ".join((s.get("text") or "").split())[:120]
        parts.append(
            f"{s.get('ordinal')}. {s.get('title') or '(untitled)'}\n{head}")
    return "\n\n".join(parts)


_UNIT_LINE_RE = re.compile(
    r"^\s*UNIT\s+([\d.]+)\s*[:.\-]\s*(.*?)\s*\|\s*SUBTOPICS\s*[:.]\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE)

_OTHER_UNIT_RE = re.compile(
    r"^\s*(?:-?\s*)?(\d+)\s*[.)]\s*(.*?)\s*[-:]\s*(.*)\s*$",
    re.MULTILINE)


def _parse_syllabus(text: str, sections: list) -> list:
    """Parse the model's outline output back into unit dicts."""
    by_ord = {s["ordinal"]: s for s in sections}
    units = []
    seen = set()
    for m in _UNIT_LINE_RE.finditer(text or ""):
        try:
            ord_raw = float(m.group(1))
        except ValueError:
            continue
        ordinal = int(ord_raw)
        if ordinal in seen:
            continue
        seen.add(ordinal)
        sec = by_ord.get(ordinal)
        if sec is None:
            continue
        subtopics = [t.strip() for t in m.group(3).split(",") if t.strip()][:5]
        title = (m.group(2).strip() or sec["title"] or f"Section {ordinal}")
        units.append({
            "unit_id": f"u{ordinal}",
            "title": title,
            "ordinal": ordinal,
            "parent_ids": [sec["parent_id"]],
            "subtopics": subtopics,
            "words": sec["words"],
        })
    # Fallback: if the model did not use the UNIT|SUBTOPICS form, try loose
    # ordinal lines, else fall back to one unit per section with no subtopics.
    if not units:
        for s in sections:
            units.append({
                "unit_id": f"u{s['ordinal']}",
                "title": s["title"] or f"Section {s['ordinal']}",
                "ordinal": s["ordinal"],
                "parent_ids": [s["parent_id"]],
                "subtopics": [],
                "words": s["words"],
            })
    units.sort(key=lambda u: u["ordinal"])
    return units


def build_syllabus(title: str, set_name: str, cfg, client) -> dict:
    """One LLM call producing a cached syllabus for the work.

    Returns {title, slug, hash, units, warnings}. On LLM failure it degrades to
    a deterministic one-unit-per-section outline so the planner never blocks.
    """
    sections = inventory(title, set_name)
    if not sections:
        return {"title": title, "slug": _slug(title), "hash": "",
                "units": [], "warnings": ["No sections found for this work."]}

    sig = quiz_store.parents_signature(sections)
    cached = quiz_store.get_cached_syllabus(set_name, _slug(title), sig)
    if cached and cached.get("units"):
        return {"title": title, "slug": _slug(title), "hash": sig,
                "units": cached["units"], "warnings": [], "cached": True}

    units = []
    warnings = []
    prompt = _build_syllabus_prompt(title, sections)
    try:
        model = cfg.get("llm_model", "default")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=min(int(cfg.get("max_tokens_summary", 4096) or 4096), 6000),
        )
        text = (resp.choices[0].message.content or "").strip()
        units = _parse_syllabus(text, sections)
        if not units:
            warnings.append("LLM outline did not parse; using section-level units.")
    except Exception as e:
        warnings.append(f"Syllabus LLM call failed ({e}); using section-level units.")
        units = _parse_syllabus("", sections)

    syllabus = {"title": title, "slug": _slug(title), "hash": sig,
                "units": units, "warnings": warnings, "cached": False}
    quiz_store.write_cached_syllabus(set_name, _slug(title), sig, syllabus)
    return syllabus


def get_syllabus(title: str, set_name: str, cfg=None, client=None) -> dict:
    """Cached read of a work's syllabus, building it on demand if absent."""
    sections = inventory(title, set_name)
    if not sections:
        return {"title": title, "units": [], "warnings": ["No sections."]}
    sig = quiz_store.parents_signature(sections)
    cached = quiz_store.get_cached_syllabus(set_name, _slug(title), sig)
    if cached and cached.get("units"):
        return {"title": title, "slug": _slug(title), "hash": sig,
                "units": cached["units"], "warnings": [], "cached": True}
    if cfg is None or client is None:
        return {"title": title, "slug": _slug(title), "hash": sig,
                "units": _parse_syllabus("", sections), "warnings": [],
                "cached": False}
    return build_syllabus(title, set_name, cfg, client)
