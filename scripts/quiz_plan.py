#!/usr/bin/env python3
"""quiz_plan.py - plan a QuizSpec from a request (work / sections / topic).

The planner is deterministic where it can be and produces an editable QuizSpec:
    spec = {
        "id": str,                 # set when persisted
        "set_name": str,
        "title": str,              # work title or topic label
        "mode": "work"|"sections"|"topic",
        "depth": "surface"|"balanced"|"deep",
        "count": int,              # total questions requested
        "difficulty": "easy"|"medium"|"hard"|"mixed",
        "types": [...],            # allowed question types
        "work": str,               # resolved work title (work-mode)
        "units": [...],            # selected units with per-unit allocation
        "warnings": [...],         # ambiguity / low-confidence notes
        "syllabus_summary": str,   # human-readable outline summary
    }

Scope resolution:
    work      -> the whole work's units, depth ignored (all units)
    sections  -> explicit unit_ids (a range or a set)
    topic     -> rank works with catalog.find_books, then rank units by cosine
                 similarity between the topic and each unit's section text using
                 the existing E5 embedder (no LLM). The depth knob decides
                 whether the topic targets whole units (surface/balanced) or
                 narrows to the strongest subtopics inside the top unit (deep).

Budget allocation distributes `count` proportional to unit word counts with a
per-unit floor (default 1). Ambiguity (multiple works matching a topic) returns
warnings; a topic that maps to zero confident units returns an explicit refusal
whose nearest works are recorded to the out-of-scope ledger.

`revise_spec` applies conversational edits (include/exclude units, change
count/difficulty/depth) and re-runs the budget allocation.
"""

import json
import math

import quiz_store
import quiz_syllabus


# ─── Depth semantics ─────────────────────────────────────────────────────────

def _unit_score_for_depth(unit: dict, depth: str, _ctx=None) -> float:
    """Score a unit for topic-scope selection (heuristic, no LLM).

    All units are treated as equally plausible prima facie; the depth knob
    mainly affects how FEW units we keep (deep = most selective) and whether we
    further narrow to subtopics. Surface/balanced keep more coverage.
    """
    words = max(unit.get("words") or 0, 1)
    base = math.log1p(words)  # longer sections are richer
    if depth == "deep":
        return base * 1.3     # favor richer sections to mine subtopics from
    if depth == "surface":
        return base * 0.7
    return base


def _top_units(ranked: list, depth: str, count: int) -> list:
    if depth == "surface":
        keep = max(1, count)
    elif depth == "deep":
        keep = max(1, math.ceil(count * 0.4))
    else:  # balanced
        keep = max(1, math.ceil(count * 0.6))
    return ranked[:keep]


# ─── Scope resolution helpers ────────────────────────────────────────────────

def _resolve_work(request, set_name, cfg) -> dict:
    """work-mode: return the whole work's units."""
    title = (request.get("work") or "").strip()
    if not title:
        return {"error": "No work title provided."}
    sections = quiz_store.inventory_sections(title, set_name)
    if not sections:
        return {"error": f"No indexed sections found for work \"{title}\"."}
    units = [{
        "unit_id": f"u{s['ordinal']}",
        "title": s["title"] or f"Section {s['ordinal']}",
        "ordinal": s["ordinal"],
        "parent_ids": [s["parent_id"]],
        "subtopics": [],
        "words": s["words"],
    } for s in sections]
    units.sort(key=lambda u: u["ordinal"])
    return {"work": title, "units": units,
            "syllabus_summary": _summarize(units), "warnings": []}


def _resolve_sections(request, set_name, cfg, syllabus) -> dict:
    """sections-mode: pick the explicit unit_ids from a work's syllabus."""
    title = syllabus.get("title", "")
    wanted = request.get("unit_ids") or request.get("sections") or []
    if not wanted:
        return {"error": "No sections/units selected."}
    if not title:
        return {"error": "A work title is required for sections mode."}
    all_units = syllabus.get("units", [])
    by_id = {u["unit_id"]: u for u in all_units}
    # Accept a contiguous range like "3-8" or a list of ids / ordinals.
    picked, errors = [], []
    for w in (wanted if isinstance(wanted, list) else [wanted]):
        w = str(w).strip()
        matched = by_id.get(w)
        if matched is None:
            # maybe an ordinal number or an ordinal->ordinal range
            if "-" in w and not w.lstrip("-").replace("-", "").isdigit():
                a, _, b = w.partition("-")
                try:
                    lo, hi = int(a), int(b)
                    lo, hi = min(lo, hi), max(lo, hi)
                    matched = [by_id.get(f"u{x}") for x in range(lo, hi + 1)
                               if by_id.get(f"u{x}")]
                    picked.extend(u for u in matched if u)
                    continue
                except ValueError:
                    pass
            errors.append(w)
            continue
        if matched not in picked:
            picked.append(matched)
    if not picked:
        return {"error": "None of the requested units matched the syllabus."}
    units = sorted(picked, key=lambda u: u["ordinal"])
    if errors:
        return {"units": units, "work": title,
                "syllabus_summary": _summarize(units),
                "warnings": [f"Unknown units ignored: {', '.join(errors)}"]}
    return {"units": units, "work": title,
            "syllabus_summary": _summarize(units), "warnings": []}


def _resolve_topic(request, set_name, cfg, embedder, collection) -> dict:
    """topic-mode: rank works, then rank their units by cosine similarity."""
    topic = (request.get("topic") or request.get("work") or "").strip()
    if not topic:
        return {"error": "No topic provided."}
    max_works = int(cfg.get("quiz_topic_max_works", 3) or 3)
    pool = int(cfg.get("quiz_topic_section_pool", 120) or 120)

    import catalog as CAT
    books = []
    if collection is not None:
        try:
            books = CAT.find_books(topic, set_name, None, cfg,
                                   collection=collection, embedder=embedder)
        except Exception as e:
            return {"error": f"Topic search failed: {e}"}
    if not books:
        books = CAT.list_books(set_name, collection)
    if not isinstance(books, list):
        books = list(books)
    works = [b["title"] for b in books[:max_works] if isinstance(b, dict) and b.get("title")]
    if not works:
        return {"error": "No works found in this set."}

    # Gather a capped pool of sections across the top works using the
    # deterministic inventory (avoid N LLM syllabus calls just to score).
    sections = []
    owners = {}
    for w in works:
        for s in quiz_store.inventory_sections(w, set_name):
            sections.append((w, s))
    if len(sections) > pool:
        sections = sections[:pool]

    # Cosine-rank sections vs the topic using the E5 embedder.
    try:
        qemb = embedder.encode([_embed_query(topic)], prompt_name="query")[0]
        texts = [_embed_passage(w + " " + s["title"] + " " + (s["text"] or ""))
                 for w, s in sections]
        embs = embedder.encode(texts, prompt_name="document")
        scored = []
        for (w, s), e in zip(sections, embs):
            sim = _cosine(qemb, e)
            scored.append((sim, w, s))
        scored.sort(key=lambda x: -x[0])
    except Exception as e:
        print(f"[quiz_plan] topic embedding failed ({e}); using inventory order",
              flush=True)
        scored = [(0.0, w, s) for w, s in sections]

    # Fold scored sections into units (one unit per section in topic mode).
    units = []
    seen = set()
    for _, w, s in scored:
        uid = f"u{s['ordinal']}"
        key = (w, s["parent_id"])
        if key in seen:
            continue
        seen.add(key)
        units.append({
            "unit_id": uid,
            "title": s["title"] or f"Section {s['ordinal']}",
            "ordinal": s["ordinal"],
            "parent_ids": [s["parent_id"]],
            "subtopics": [],
            "words": s["words"],
            "work": w,
        })
    if not units:
        quiz_store.append_out_of_scope({
            "set_name": set_name, "type": "topic", "query": topic,
            "reason": "topic mapped to no units",
        })
        return {"refusal": True,
                "message": (f"Your topic \"{topic}\" did not map to any "
                            "confident section of the library."),
                "nearest_works": works[:5], "units": [], "warnings": [],
                "syllabus_summary": ""}

    # Open questions: topic may match multiple works.
    owners_found = sorted({u["work"] for u in units})
    warnings = []
    if len(owners_found) > 1:
        warnings.append(f"Topic matches multiple works: {', '.join(owners_found)}.")
    depth = request.get("depth") or "balanced"
    keep = _top_units(units, depth, int(cfg.get("quiz_default_count", 10) or 10))
    units = keep
    summary = _summarize([{**u, "title": f"{u['work']} — {u['title']}"}
                          for u in units])
    return {"work": ", ".join(owners_found), "units": units,
            "syllabus_summary": summary, "warnings": warnings}


# ─── Embedding helpers (E5 prefixes) ─────────────────────────────────────────

def _embed_query(topic: str) -> str:
    return f"query: {topic}"


def _embed_passage(text: str) -> str:
    return f"passage: {text[:1500]}"


def _cosine(a, b) -> float:
    a = a.astype(float).flatten()
    b = b.astype(float).flatten()
    import numpy as _np
    na = _np.linalg.norm(a)
    nb = _np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(_np.dot(a, b) / (na * nb))


# ─── Budget allocation ───────────────────────────────────────────────────────

def _project(allocated: list) -> list:
    """Pick the serializable unit fields (never includes the full text)."""
    keys = ("unit_id", "title", "ordinal", "parent_ids", "subtopics", "words",
            "selected", "allocation")
    return [{k: u.get(k) for k in keys if k in u} for u in allocated]


def _title_key(s: str) -> str:
    """Lightweight title normalization shared with quiz_store.weak_unit_signal."""
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _weakness_weights(set_name: str) -> dict:
    """Return a {normalized unit-title: weight} map from stored analytics.

    SESSION 5 weakness-weighted builder defaults: units that were repeatedly
    missed in past finished attempts get a weight > 1 so the next plan proposes
    more questions for them. Absent signal yields weight 1.0.
    """
    signal = quiz_store.weak_unit_signal(set_name)
    if not signal:
        return {}
    max_miss = max(signal.values())
    weights = {}
    for key, misses in signal.items():
        # 1 + misses/max_miss → weak units land in roughly [1, 2].
        weights[key] = 1.0 + (misses / max_miss)
    return weights


def allocate_budget(units: list, count: int, floor: int = 1,
                    weights: dict | None = None) -> list:
    """Distribute `count` questions across units, weighted by word counts and
    (optionally) weakness weights from stored analytics.

    Each selected unit gets at least `floor`; the remainder is distributed
    proportional to ``words * weight``. ``weights`` maps a normalized unit title
    to a multiplier (SESSION 5 weakness-weighted defaults). Returns a new list
    of units with an ``allocation`` field.
    """
    if not units:
        return []
    count = max(0, int(count))
    weights = weights or {}
    sel = [u for u in units if u.get("selected", True)]
    if not sel:
        sel = list(units)

    def _weight(u) -> float:
        return max(1.0, float(weights.get(_title_key(u.get("title")), 1.0)))

    total_w = sum(max(u.get("words") or 0, 1) * _weight(u) for u in sel)
    out = []
    for u in sel:
        u = dict(u)
        if count <= 0:
            u["allocation"] = 0
        else:
            w = max(u.get("words") or 0, 1) * _weight(u)
            alloc = int(count * w / total_w)
            u["allocation"] = max(floor, alloc)
        out.append(u)
    # Rebalance: enforce the total (floor may over-allocate).
    residual = count - sum(u["allocation"] for u in out)
    if residual < 0:
        # Trim from the largest allocations first, keeping >= floor.
        idx = sorted(range(len(out)), key=lambda i: -out[i]["allocation"])
        for i in idx:
            if residual >= 0:
                break
            if out[i]["allocation"] > floor:
                out[i]["allocation"] -= 1
                residual += 1
    elif residual > 0:
        # Add to the largest by weight, cycling.
        order = sorted(range(len(out)), key=lambda i: -(
            max(out[i].get("words") or 0, 1) * _weight(out[i])))
        j = 0
        while residual > 0:
            i = order[j % len(order)]
            out[i]["allocation"] += 1
            residual -= 1
            j += 1
    return out


# ─── Public API ──────────────────────────────────────────────────────────────

def _summarize(units: list) -> str:
    if not units:
        return "No units."
    parts = []
    for u in units[:8]:
        label = u.get("title") or f"Unit {u.get('ordinal')}"
        sub = u.get("subtopics") or []
        if sub:
            label += " — " + ", ".join(sub[:3])
        parts.append(label)
    more = len(units) - len(parts)
    if more > 0:
        parts.append(f"… and {more} more")
    return "; ".join(parts)


def _request_types(request, cfg) -> list:
    types = request.get("types")
    if types:
        return [t for t in types if t in (
            "mcq", "short_answer", "true_false", "fill_blank")]
    return ["mcq", "true_false", "short_answer", "fill_blank"]


def plan_quiz(request, set_name, cfg, embedder=None, collection=None,
              persist: bool = True) -> dict:
    """Resolve a quiz request into an editable QuizSpec.

    `request` is one of:
        {"mode": "work",    "work": "<title>", ...}
        {"mode": "sections","work": "<title>", "unit_ids": [...]|"3-8", ...}
        {"mode": "topic",   "topic": "<text>", ...}
    Common keys: count, difficulty, types, depth, minimum_per_unit.
    """
    mode = (request.get("mode") or "topic").strip().lower()
    if mode not in ("work", "sections", "topic"):
        mode = "topic"
    depth = (request.get("depth") or "balanced").strip().lower()
    if depth not in ("surface", "balanced", "deep"):
        depth = "balanced"
    count = int(cfg.get("quiz_default_count", 10) or 10)
    if request.get("count") is not None:
        count = int(request.get("count") or count)
    count = max(1, min(count, int(cfg.get("quiz_max_batches", 12) or 12) * 200 or 1200))
    floor = max(1, int(request.get("minimum_per_unit", 1) or 1))

    # syllabus used for sections/work resolution.
    syllabus = {}
    if mode in ("work", "sections"):
        # work mode: build units from inventory deterministically (no LLM
        # outline needed for planning); sections mode uses get_syllabus.
        if mode == "work":
            resolved = _resolve_work(request, set_name, cfg)
        else:
            title = (request.get("work") or "").strip()
            if not title:
                return {"error": "A work title is required for sections mode."}
            syllabus = quiz_syllabus.get_syllabus(title, set_name)
            resolved = _resolve_sections(request, set_name, cfg, syllabus)
    else:
        resolved = _resolve_topic(request, set_name, cfg, embedder, collection)

    if "error" in resolved:
        return {"error": resolved["error"]}
    if resolved.get("refusal"):
        return resolved

    units = resolved["units"]
    # SESSION 5 weakness-weighted builder defaults: propose more questions for
    # units that were repeatedly missed in past attempts (user can still
    # override each unit's allocation in the spec card).
    weights = _weakness_weights(set_name)
    allocated = allocate_budget(units, count, floor, weights=weights)
    for u in allocated:
        u["selected"] = True

    weakness_weighted = [
        u["title"] for u in allocated
        if (weights or {}).get(_title_key(u.get("title")), 1.0) > 1.0
    ]
    warnings = list(resolved.get("warnings") or [])
    if weakness_weighted:
        warnings.append(
            "Focused more questions on prior weak areas: "
            + ", ".join(weakness_weighted[:5])
            + (" …" if len(weakness_weighted) > 5 else "")
            + " (edit any unit's count to override).")

    spec = {
        "set_name": set_name,
        "title": resolved.get("work") or resolved.get("title") or "Quiz",
        "mode": mode,
        "depth": depth,
        "count": count,
        "difficulty": request.get("difficulty") or "mixed",
        "types": _request_types(request, cfg),
        "work": resolved.get("work", ""),
        "units": _project(allocated),
        "warnings": warnings,
        "syllabus_summary": resolved.get("syllabus_summary") or "",
        "minimum_per_unit": floor,
    }
    if persist:
        spec = quiz_store.create_spec(spec)
    return {"spec": spec, "warnings": spec["warnings"],
            "out_of_scope": False, "syllabus_summary": spec["syllabus_summary"]}


def revise_spec(spec: dict, deltas: dict) -> dict:
    """Apply conversational edits and re-run the budget allocation."""
    if not spec:
        return {"error": "No spec to revise."}
    spec = dict(spec)
    units = [dict(u) for u in spec.get("units", [])]

    # include / exclude unit_ids
    exclude = deltas.get("exclude") or deltas.get("exclude_units")
    if exclude:
        exclude = {str(u) for u in (exclude if isinstance(exclude, list)
                                    else [exclude])}
        for u in units:
            if u["unit_id"] in exclude:
                u["selected"] = False
    include = deltas.get("include") or deltas.get("include_units")
    if include:
        include = {str(u) for u in (include if isinstance(include, list)
                                    else [include])}
        for u in units:
            if u["unit_id"] in include:
                u["selected"] = True

    if "count" in deltas and deltas["count"] is not None:
        spec["count"] = max(1, int(deltas["count"]))
    if "difficulty" in deltas and deltas["difficulty"]:
        spec["difficulty"] = deltas["difficulty"]
    if "depth" in deltas and deltas["depth"] in ("surface", "balanced", "deep"):
        spec["depth"] = deltas["depth"]
    if "types" in deltas and isinstance(deltas["types"], list):
        spec["types"] = [t for t in deltas["types"] if t in (
            "mcq", "short_answer", "true_false", "fill_blank")]
    if "title" in deltas and deltas["title"]:
        spec["title"] = deltas["title"]

    floor = max(1, int(spec.get("minimum_per_unit", 1) or 1))
    if "counts" in deltas and isinstance(deltas["counts"], dict):
        # Manual per-unit allocation overrides (e.g. from the spec card).
        alloc_map = {str(k): int(v) for k, v in deltas["counts"].items()
                     if v is not None}
        for u in units:
            if u["unit_id"] in alloc_map:
                u["allocation"] = max(0, alloc_map[u["unit_id"]])
        spec["count"] = sum(u.get("allocation", 0) for u in units
                            if u.get("selected", True))
        allocated = units
    else:
        # Re-apply weakness weights unless the user is hand-tuning the total.
        weights = _weakness_weights(spec.get("set_name") or "")
        allocated = allocate_budget(units, spec.get("count", 10), floor,
                                    weights=weights)
    spec["units"] = _project(allocated)
    spec["syllabus_summary"] = _summarize(spec["units"])
    if spec.get("id"):
        spec = quiz_store.update_spec(spec)
    return {"spec": spec, "warnings": spec.get("warnings", []),
            "out_of_scope": False, "syllabus_summary": spec["syllabus_summary"]}
