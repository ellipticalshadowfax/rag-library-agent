#!/usr/bin/env python3
"""quiz_store.py - JSON persistence for quiz specs, generated quizzes, cached
syllabi, and the out-of-scope ledger.

Mirrors chat_store.py: each quiz / syllabus / ledger entry is a JSON file (or a
key inside a shared file) under <rag_root>/quizzes/. The directory is
gitignored and never committed. All writes are atomic (tmp file + os.replace)
so a crash never leaves a truncated file. IDs are 12-hex like chat_store.

Layout under <rag_root>/quizzes/:
    quizzes/<id>.json                    generated quiz artifacts (SESSION 3+)
    quizzes/specs/<id>.json              editable quiz specs from the planner
    quizzes/syllabus/<set>/<slug>.json   cached LLM syllabi, keyed to a hash of
                                         the parents table rows for the work
    quizzes/out_of_scope.json            ledger of refused topic plans
    quizzes/review.json                  shared FSRS review store (SESSION 5)

SESSION 2 only covers specs, syllabus cache and the out-of-scope ledger.
Generated-quiz files (<id>.json) gain their full shape in SESSION 3; the CRUD
helpers for them exist here now so the store is stable.
"""

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from _paths import rag_root

QUIZ_DIR = rag_root() / "quizzes"


# ─── Path helpers ────────────────────────────────────────────────────────────

def _spec_path(qid: str) -> Path:
    return QUIZ_DIR / "specs" / f"{qid}.json"


def _quiz_path(qid: str) -> Path:
    return QUIZ_DIR / f"{qid}.json"


def _syllabus_dir(set_name: str) -> Path:
    return QUIZ_DIR / "syllabus" / _safe(set_name)


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s) or "_"


def _now() -> int:
    return int(time.time())


def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ─── Out-of-scope ledger ─────────────────────────────────────────────────────

def append_out_of_scope(entry: dict) -> None:
    """Append a refused/ambiguous topic plan to the ledger."""
    ledger = _read_ledger()
    ledger.append({
        "ts": _now(),
        **(entry or {}),
    })
    _atomic_write(QUIZ_DIR / "out_of_scope.json", ledger)


def read_out_of_scope(limit: int = 200) -> list:
    return list(reversed(_read_ledger()))[:limit]


def _read_ledger() -> list:
    p = QUIZ_DIR / "out_of_scope.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ─── Quiz specs (SESSION 2) ──────────────────────────────────────────────────

def create_spec(spec: dict) -> dict:
    """Persist a quiz spec and return it with an id + timestamps."""
    QUIZ_DIR.mkdir(exist_ok=True)
    (QUIZ_DIR / "specs").mkdir(exist_ok=True)
    now = _now()
    spec.setdefault("id", uuid.uuid4().hex[:12])
    spec.setdefault("created", now)
    spec["updated"] = now
    _atomic_write(_spec_path(spec["id"]), spec)
    return spec


def update_spec(spec: dict) -> dict:
    spec = dict(spec)
    spec["updated"] = _now()
    _atomic_write(_spec_path(spec["id"]), spec)
    return spec


def get_spec(qid: str) -> dict | None:
    p = _spec_path(qid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_specs(set_name: str | None = None) -> list:
    d = QUIZ_DIR / "specs"
    if not d.exists():
        return []
    out = []
    for p in d.glob("*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            if set_name and s.get("set_name") != set_name:
                continue
            out.append({
                "id": s.get("id"),
                "title": s.get("title", "Untitled"),
                "mode": s.get("mode", "work"),
                "set_name": s.get("set_name", ""),
                "count": s.get("count", 0),
                "unit_count": len(s.get("units", [])),
                "created": s.get("created", 0),
                "updated": s.get("updated", 0),
            })
        except Exception:
            continue
    out.sort(key=lambda s: s.get("updated", 0), reverse=True)
    return out


def delete_spec(qid: str) -> bool:
    p = _spec_path(qid)
    if p.exists():
        p.unlink()
        return True
    return False


# ─── Cached syllabi (SESSION 2) ──────────────────────────────────────────────

def _syllabus_path(set_name: str, slug: str, parents_hash: str) -> Path:
    return _syllabus_dir(set_name) / f"{slug}.{parents_hash}.json"


def get_cached_syllabus(set_name: str, slug: str, parents_hash: str):
    p = _syllabus_path(set_name, slug, parents_hash)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_cached_syllabus(set_name: str, slug: str, parents_hash: str,
                          syllabus: dict) -> dict:
    syllabus["set_name"] = set_name
    syllabus["slug"] = slug
    syllabus["parents_hash"] = parents_hash
    syllabus["cached_at"] = _now()
    _atomic_write(_syllabus_path(set_name, slug, parents_hash), syllabus)
    return syllabus


# ─── Generated quizzes (SESSION 3, shape defined there) ──────────────────────

def create_quiz(q: dict) -> dict:
    QUIZ_DIR.mkdir(exist_ok=True)
    q.setdefault("id", uuid.uuid4().hex[:12])
    q.setdefault("created", _now())
    q["updated"] = _now()
    _atomic_write(_quiz_path(q["id"]), q)
    return q


def update_quiz(q: dict) -> dict:
    q = dict(q)
    q["updated"] = _now()
    _atomic_write(_quiz_path(q["id"]), q)
    return q


def get_quiz(qid: str) -> dict | None:
    p = _quiz_path(qid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_quizzes(set_name: str | None = None) -> list:
    out = []
    for p in QUIZ_DIR.glob("*.json"):
        if p.name == "out_of_scope.json" or p.name == "review.json":
            continue
        try:
            q = json.loads(p.read_text(encoding="utf-8"))
            if set_name and q.get("set_name") != set_name:
                continue
            out.append({
                "id": q.get("id"),
                "title": q.get("title", "Untitled"),
                "set_name": q.get("set_name", ""),
                "status": q.get("status", "unknown"),
                "question_count": len(q.get("questions", [])),
                "created": q.get("created", 0),
                "updated": q.get("updated", 0),
            })
        except Exception:
            continue
    out.sort(key=lambda q: q.get("updated", 0), reverse=True)
    return out


def delete_quiz(qid: str) -> bool:
    p = _quiz_path(qid)
    if p.exists():
        p.unlink()
        return True
    return False


# ─── Shared helpers used by the planner ──────────────────────────────────────

def inventory_sections(title: str, set_name: str) -> list:
    """Deterministic section inventory of a work from manifest.db `parents`.

    Returns ordered sections: [{parent_id, title, ordinal, words}].
    Mirrors catalog.list_sections but also exposes the parent_id and text
    length (word count) needed for the syllabus / budget allocation.
    """
    db = rag_root() / "manifest.db"
    if not db.exists() or not title:
        return []
    try:
        con = sqlite3.connect(str(db))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT parent_id, section_title, section_ordinal, text FROM parents "
            "WHERE set_name=? AND title=? ORDER BY section_ordinal",
            (set_name, title)).fetchall()
        con.close()
    except Exception as e:
        print(f"[quiz_store] section inventory failed: {e}", flush=True)
        return []
    sections = []
    for r in rows:
        text = r["text"] or ""
        sections.append({
            "parent_id": r["parent_id"] or "",
            "title": r["section_title"] or "",
            "ordinal": r["section_ordinal"],
            "words": len(text.split()),
            "text": text,
        })
    return sections


def parents_signature(sections: list) -> str:
    """Hash the inventory rows so re-ingest (ordinal/parent/text change)
    invalidates a cached syllabus."""
    import hashlib
    h = hashlib.sha256()
    for s in sections:
        h.update(f"{s['ordinal']}|{s['parent_id']}|{s['words']}|{s['title']}\n".encode())
    return h.hexdigest()[:16]


# ─── FSRS review store (SESSION 5) ───────────────────────────────────────────
#
# A single shared ``quizzes/review.json`` keyed by ``<quiz_id>:<qid>`` so the
# due-today queue spans quizzes (SESSION 4's missed/weak questions get enrolled
# here so they resurface on an FSRS schedule). Each card snapshots its own
# question (with the answer, so the player can self-check), carries the serial
# ``fsrs`` card (round-trippable via fsrs.Card.from_dict / to_dict), and a small
# ``history`` of every review (rating / ts / stability) used to render a
# strength-over-time view. Writes are atomic like every other store entry.
#
# Layout:
#     quizzes/review.json
#         {"cards": {"<quiz_id>:<qid>": {
#             "quiz_id", "qid", "set_name", "type",
#             "question": {...full snapshot incl. answer...},
#             "fsrs": {...fsrs.Card.to_dict()...},
#             "enrolled_at", "last_rating", "last_rated_at",
#             "history": [{"rating", "ts", "stability", "difficulty"}]
#         }}}

_RATING_MAP = {"again": 1, "hard": 2, "good": 3, "easy": 4}  # fsrs.Rating ints


def _review_path() -> Path:
    return QUIZ_DIR / "review.json"


def _read_review_store() -> dict:
    p = _review_path()
    if not p.exists():
        return {"cards": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("cards"), dict):
            return data
    except Exception:
        pass
    return {"cards": {}}


def _write_review_store(data: dict) -> None:
    _atomic_write(_review_path(), data)


def _review_key(quiz_id: str, qid: str) -> str:
    return f"{quiz_id}:{qid}"


def review_card_key(quiz_id: str, qid: str) -> str:
    """The composite key a review card is stored under (exposed for the API)."""
    return _review_key(quiz_id, qid)


def enroll_review_question(quiz_id: str, set_name: str, qid: str,
                           question: dict, already_correct: bool = False) -> dict | None:
    """Enroll a question into the shared review store as a new FSRS card.

    Enrolling a question the learner already got right on the attempt starts it
    in a strong-ish state (Good) so the first review is not due immediately; a
    missed question starts un-reviewed (Learning) so it comes due right away.
    Idempotent: re-enrolling the same (quiz_id, qid) is a no-op.
    Returns the (possibly newly created) card dict, or None if FSRS is
    unavailable / the question has no ``q`` text.
    """
    if not (question or {}).get("q"):
        return None
    store = _read_review_store()
    key = _review_key(quiz_id, qid)
    if key in store["cards"]:
        return store["cards"][key]
    try:
        from fsrs import Card, Rating, Scheduler
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    card = Card()
    scheduler = Scheduler()
    if already_correct:
        card, _log = scheduler.review_card(card, Rating.Good, now)
    store["cards"][key] = {
        "quiz_id": quiz_id,
        "qid": str(qid),
        "set_name": set_name,
        "type": (question.get("type") or "mcq"),
        "question": question,
        "fsrs": card.to_dict(),
        "enrolled_at": int(time.time()),
        "last_rating": "good" if already_correct else None,
        "last_rated_at": int(time.time()) if already_correct else None,
        "history": [{
            "rating": "good",
            "ts": int(time.time()),
            "stability": card.stability,
            "difficulty": card.difficulty,
        }] if already_correct else [],
    }
    _write_review_store(store)
    return store["cards"][key]


def get_review_card(quiz_id: str, qid: str) -> dict | None:
    return _read_review_store()["cards"].get(_review_key(quiz_id, qid))


def _parse_fsrs_due(due) -> datetime | None:
    """Parse an fsrs ISO due string (UTC, with offset) to a tz-aware datetime."""
    if not due:
        return None
    if isinstance(due, datetime):
        return due if due.tzinfo else due.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(due))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def record_review_rating(quiz_id: str, qid: str, rating: str) -> dict | None:
    """Apply an FSRS rating (again|hard|good|easy) to a stored review card.

    Recomputes the card schedule with the current UTC time, records the review
    in the card's history (with the new stability for the strength-over-time
    view), and persists atomically. Returns the updated card, or None if the
    card is not enrolled or the rating is invalid.
    """
    norm = (rating or "").strip().lower()
    if norm not in _RATING_MAP:
        return None
    store = _read_review_store()
    key = _review_key(quiz_id, qid)
    entry = store["cards"].get(key)
    if not entry:
        return None
    try:
        from fsrs import Card, Rating, Scheduler
    except Exception:
        return None
    try:
        card = Card.from_dict(entry.get("fsrs") or {})
        rating_enum = Rating(_RATING_MAP[norm])
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    try:
        new_card, _log = Scheduler().review_card(card, rating_enum, now)
    except Exception:
        return None
    entry["fsrs"] = new_card.to_dict()
    entry["last_rating"] = norm
    entry["last_rated_at"] = int(time.time())
    (entry.setdefault("history", [])).append({
        "rating": norm,
        "ts": int(time.time()),
        "stability": new_card.stability,
        "difficulty": new_card.difficulty,
    })
    _write_review_store(store)
    return entry


def delete_review_card(quiz_id: str, qid: str) -> bool:
    store = _read_review_store()
    key = _review_key(quiz_id, qid)
    if key not in store["cards"]:
        return False
    del store["cards"][key]
    _write_review_store(store)
    return True


def fetch_due_reviews(set_name: str | None = None, limit: int = 50,
                      quiz_ids: set | None = None,
                      now: datetime | None = None) -> list:
    """Cards whose FSRS ``due`` time has passed, oldest-due first.

    ``set_name`` filters to one collection; ``quiz_ids`` narrows to a specific
    set of quiz ids (used by the due-badge per quiz). ``now`` is injectable for
    tests. Each returned dict is the stored card entry plus its ``key``.
    """
    now = now or datetime.now(timezone.utc)
    store = _read_review_store()
    out = []
    for key, entry in store["cards"].items():
        if set_name and entry.get("set_name") != set_name:
            continue
        if quiz_ids is not None and entry.get("quiz_id") not in quiz_ids:
            continue
        due = _parse_fsrs_due((entry.get("fsrs") or {}).get("due"))
        if due is None or due <= now:
            out.append({**entry, "key": key})
    out.sort(key=lambda e: _parse_fsrs_due((e.get("fsrs") or {}).get("due"))
             or datetime.min.replace(tzinfo=timezone.utc))
    return out[:limit]


def fetch_review_stats(set_name: str | None = None) -> dict:
    """Aggregate review-card statistics: counts, retention, and a small
    strength-over-time series (average stability per day) for the Review pane."""
    store = _read_review_store()
    cards = [e for e in store["cards"].values()
             if not set_name or e.get("set_name") == set_name]
    now = datetime.now(timezone.utc)
    due_count = 0
    new_count = 0
    passes = 0
    rated = 0
    strength_buckets = {}
    for e in cards:
        if not (e.get("history") or []):
            new_count += 1
        else:
            rated += 1
            if e.get("last_rating") in ("good", "easy"):
                passes += 1
        due = _parse_fsrs_due((e.get("fsrs") or {}).get("due"))
        if due is None or due <= now:
            due_count += 1
        # Strength-over-time: bucket average stability by review day.
        for h in e.get("history") or []:
            day = h.get("ts")
            if not day:
                continue
            bucket = strength_buckets.setdefault(day // 86400, {"sum": 0.0, "n": 0})
            if isinstance(h.get("stability"), (int, float)):
                bucket["sum"] += float(h.get("stability"))
                bucket["n"] += 1
    retention = round(passes / rated, 4) if rated else 0.0
    series = [{"day": day * 86400, "stability": round(b["sum"] / b["n"], 3)}
              for day, b in sorted(strength_buckets.items()) if b["n"] > 0]
    return {
        "total_cards": len(cards),
        "due_count": due_count,
        "new_count": new_count,
        "reviewed_count": rated,
        "retention_rate": retention,
        "strength_over_time": series,
    }


def _title_key(s: str) -> str:
    """Lightweight normalization for matching plan units to analytics units."""
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def weak_unit_signal(set_name: str | None = None) -> dict:
    """Scan generated quizzes' finished attempts for weakly-mastered units and
    return a {normalized section title: miss_count} map (SESSION 5
    weakness-weighted builder defaults).

    Each ``study_points`` entry in an attempt's analytics carries a section
    title; counts accumulate across ALL finished attempts so a unit that is
    repeatedly missed inflates its weight for the next plan. The planner uses
    this to propose more questions for weak units (the user can still override
    the per-unit allocation in the spec card).
    """
    counts: dict = {}
    for p in QUIZ_DIR.glob("*.json"):
        if p.name in ("out_of_scope.json", "review.json"):
            continue
        try:
            quiz = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if set_name and quiz.get("set_name") != set_name:
            continue
        for attempt in quiz.get("attempts") or []:
            an = attempt.get("analytics") or {}
            for sp in an.get("study_points") or []:
                key = sp.get("section_title") or sp.get("unit")
                if key:
                    counts[_title_key(key)] = counts.get(_title_key(key), 0) + 1
    return counts
