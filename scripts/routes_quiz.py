"""Quiz, review and course routes.

Registered on the app by server.py. The origin guard stays on the app.
"""

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request, send_from_directory, Response
from _paths import (LOCAL_KEYS, SECRET_KEYS, atomic_write_json, cuda_available,
                    rag_root, resolve_device)
import agent
from agent import CollectionNotFound
import chat_store
import ingest as ING
import scan as SCAN
import catalog as CAT
import study as STUDY
import quiz_build
import quiz_export
import quiz_grade
import quiz_plan
import quiz_syllabus
import quiz_store
from flask import Blueprint

from server_core import (
    _as_int,
    _course_quizzed_counts,
    _quiz_attempt_id,
    _quiz_generate_sse,
    _quiz_plan_collection,
    _set_name_from_request,
    _strip_answers,
    cfg_float,
    cfg_int,
    get_chat_client,
    get_chat_collection,
    get_chat_embedder,
    ingest_active,
    load_cfg,
    map_reduce_summary,
)

bp = Blueprint("quiz", __name__)


@bp.route("/api/quiz/plan", methods=["POST"])
def api_quiz_plan():
    """Resolve a quiz request (work / sections / topic) into an editable spec."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or _set_name_from_request()
    cfg = load_cfg()
    collection, embedder = _quiz_plan_collection(set_name)
    try:
        result = quiz_plan.plan_quiz(
            data, set_name, cfg, embedder=embedder, collection=collection)
    except Exception as e:
        print(f"[quiz/plan] failed: {e}", flush=True)
        return jsonify({"error": f"Quiz planning failed: {e}"}), 500
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)


@bp.route("/api/quiz/plan/revise", methods=["POST"])
def api_quiz_plan_revise():
    """Apply conversational edits to a stored quiz spec (include/exclude/count…)."""
    data = request.get_json(force=True, silent=True) or {}
    spec_id = data.get("spec_id") or (data.get("spec") or {}).get("id")
    spec = data.get("spec")
    if spec is None and spec_id:
        spec = quiz_store.get_spec(spec_id)
    if not spec:
        return jsonify({"error": "No spec provided."}), 400
    deltas = data.get("deltas") or data
    try:
        result = quiz_plan.revise_spec(spec, deltas)
    except Exception as e:
        print(f"[quiz/plan/revise] failed: {e}", flush=True)
        return jsonify({"error": f"Spec revision failed: {e}"}), 500
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)


@bp.route("/api/quiz/syllabus", methods=["POST"])
def api_quiz_syllabus():
    """Build/return a work's cached syllabus (for the Study-tab builder)."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "library"
    title = (data.get("work") or data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "No work title provided."}), 400
    cfg = load_cfg()
    try:
        syllabus = quiz_syllabus.get_syllabus(
            title, set_name, cfg, get_chat_client())
    except Exception as e:
        print(f"[quiz/syllabus] failed: {e}", flush=True)
        return jsonify({"error": f"Syllabus failed: {e}"}), 500
    return jsonify(syllabus)


@bp.route("/api/quiz/specs", methods=["GET"])
def api_quiz_specs():
    set_name = request.args.get("set_name") or "library"
    return jsonify({"specs": quiz_store.list_specs(set_name)})


@bp.route("/api/quiz/spec/<spec_id>", methods=["GET"])
def api_quiz_spec_get(spec_id):
    spec = quiz_store.get_spec(spec_id)
    if not spec:
        return jsonify({"error": "Spec not found."}), 404
    return jsonify(spec)


@bp.route("/api/quiz/spec/<spec_id>", methods=["DELETE"])
def api_quiz_spec_delete(spec_id):
    quiz_store.delete_spec(spec_id)
    return jsonify({"ok": True})


@bp.route("/api/quiz/out-of-scope", methods=["GET"])
def api_quiz_out_of_scope():
    return jsonify({"entries": quiz_store.read_out_of_scope()})


@bp.route("/api/quiz/books", methods=["GET"])
def api_quiz_books():
    """List the works in a set (titles) for the Study-tab work picker."""
    set_name = request.args.get("set_name") or "library"
    collection = None
    try:
        collection = get_chat_collection(set_name)
    except (SystemExit, CollectionNotFound):
        pass
    if collection is None or collection.count() == 0:
        return jsonify({"books": []})
    books = CAT.list_books(set_name, collection)
    return jsonify({"books": [{"title": b["title"], "kind": b.get("kind", "")}
                               for b in books if b.get("title")]})


# ─── Quiz generation pipeline (SESSION 3) ───────────────────────────────────
#
# Work-specific generation: material comes from an ordered walk of the
# `parents` table (never semantic search). Each batch is generated and put
# through the 4-stage audit (scripts/quiz_audit.py) — parse sanity, grounding
# against the real parent text, inline LLM answer-key verification, and cross-
# batch dedupe — with drop + top-up. The build runs in a worker thread and
# streams SSE ``progress`` (heartbeat per unit) and ``delta`` ({type: unit}/
# {type: done}) events via a queue, mirroring the chat stream pattern.
# Generation is blocked while an ingest is running (same reason as /api/chat:
# avoid ChromaDB concurrent read/write errors).

@bp.route("/api/quiz/generate", methods=["POST"])
def api_quiz_generate():
    """Generate a quiz from a persisted QuizSpec. SSE progress events."""
    if ingest_active():
        return jsonify({"error": "An ingest is running; wait for it to finish "
                                 "before generating a quiz."}), 409
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "library"
    spec_id = data.get("spec_id")
    spec = data.get("spec")
    if spec is None and spec_id:
        spec = quiz_store.get_spec(spec_id)
    if not spec:
        return jsonify({"error": "No spec provided."}), 400
    cfg = load_cfg()
    return Response(
        _quiz_generate_sse(set_name, spec, cfg),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no"},
    )


@bp.route("/api/quizzes", methods=["GET"])
def api_quizzes_list():
    """List generated quizzes (status, counts) for the Study tab."""
    set_name = request.args.get("set_name") or "library"
    return jsonify({"quizzes": quiz_store.list_quizzes(set_name)})


@bp.route("/api/quiz/<quiz_id>", methods=["GET"])
def api_quiz_get(quiz_id):
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    return jsonify(quiz)


@bp.route("/api/quiz/<quiz_id>/export", methods=["GET"])
def api_quiz_export(quiz_id):
    """Export a stored quiz in a portable format (markdown|gift|csv|json|apkg).

    Wires the `quiz_output_format` config key as the default when no
    ``format`` query param is given (SESSION 6)."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    fmt = request.args.get("format") or str(
        load_cfg().get("quiz_output_format", "markdown") or "markdown")
    try:
        payload, mime, is_binary = quiz_export.export_quiz(quiz, fmt)
    except ImportError as e:
        return jsonify({"error": str(e)}), 400
    fname = f"quiz-{quiz.get('id', quiz_id)}.{'apkg' if fmt == 'apkg' else 'gift' if fmt == 'gift' else 'csv' if fmt == 'csv' else 'json' if fmt == 'json' else 'md'}"
    return Response(payload, mimetype=mime,
                    headers={"Content-Disposition":
                             f'attachment; filename="{fname}"'})


@bp.route("/api/quiz/<quiz_id>", methods=["DELETE"])
def api_quiz_delete(quiz_id):
    quiz_store.delete_quiz(quiz_id)
    return jsonify({"ok": True})


@bp.route("/api/quiz/<quiz_id>/attempt", methods=["POST"])
def api_quiz_attempt(quiz_id):
    """Start a quiz attempt: register it and return the questions sans answers."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    questions = [q for q in (quiz.get("questions") or []) if q.get("q")]
    if not questions:
        return jsonify({"error": "This quiz has no questions."}), 400
    attempt_id = _quiz_attempt_id()
    attempts = quiz.setdefault("attempts", [])
    attempts.append({
        "attempt_id": attempt_id,
        "started": int(time.time()),
        "grades": {},
        "finished": False,
        "analytics": None,
    })
    quiz_store.update_quiz(quiz)
    return jsonify({
        "attempt_id": attempt_id,
        "total": len(questions),
        "questions": [_strip_answers(q) for q in questions],
    })


@bp.route("/api/quiz/<quiz_id>/answer", methods=["POST"])
def api_quiz_answer(quiz_id):
    """Grade a single answer and store it in the current attempt."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    data = request.get_json(force=True, silent=True) or {}
    attempt_id = data.get("attempt_id")
    raw_qid = data.get("qid")
    response = data.get("response")
    if not attempt_id or raw_qid is None or str(raw_qid) == "":
        return jsonify({"error": "attempt_id and qid are required."}), 400
    qid = str(raw_qid)
    attempts = quiz.get("attempts") or []
    attempt = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
    if not attempt:
        return jsonify({"error": "Attempt not found."}), 404
    question = next((q for q in (quiz.get("questions") or [])
                     if str(q.get("qid")) == qid), None)
    if not question:
        return jsonify({"error": "Question not found."}), 404
    cfg = load_cfg()
    graded = quiz_grade.grade_answer(question, response, cfg, get_chat_client())
    if graded is None:
        return jsonify({"error": "Could not grade this answer."}), 400
    attempt.setdefault("grades", {})[qid] = {
        "response": response,
        "correct": graded["correct"],
        "expected": graded["expected"],
        "feedback": graded["feedback"],
        "score": graded["score"],
        "ts": int(time.time()),
    }
    quiz_store.update_quiz(quiz)
    return jsonify({
        "correct": graded["correct"],
        "expected": graded["expected"],
        "feedback": graded["feedback"],
        "score": graded["score"],
    })


@bp.route("/api/quiz/<quiz_id>/finish", methods=["POST"])
def api_quiz_finish(quiz_id):
    """Finish an attempt: compute analytics and persist the attempt."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    data = request.get_json(force=True, silent=True) or {}
    attempt_id = data.get("attempt_id")
    if not attempt_id:
        return jsonify({"error": "attempt_id is required."}), 400
    attempts = quiz.get("attempts") or []
    attempt = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
    if not attempt:
        return jsonify({"error": "Attempt not found."}), 404
    analytics = quiz_grade.compute_analytics(
        attempt.get("grades") or {}, quiz.get("questions") or [],
        weak_threshold=cfg_float(load_cfg(), "quiz_weak_threshold", 0.6, 0, 1))
    # Enroll missed questions into the FSRS review store (SESSION 5) so they
    # resurface on a spaced-repetition schedule in the Review pane. Gated by
    # the fsrs_enabled config key; idempotent per (quiz_id, qid).
    enrolled = 0
    if load_cfg().get("fsrs_enabled", True):
        # Mirror quiz_grade.enumerate_all's qid rule (qid else array index) so
        # study points line back up even for legacy questions without a qid.
        by_qid = {str(q.get("qid") or i): q
                  for i, q in enumerate(quiz.get("questions") or [])}
        for sp in analytics.get("study_points") or []:
            qid = str(sp.get("qid"))
            q = by_qid.get(qid)
            if q and quiz_store.enroll_review_question(
                    quiz_id, quiz.get("set_name") or "", qid, q,
                    already_correct=False):
                enrolled += 1
    attempt["finished"] = True
    attempt["analytics"] = analytics
    attempt["finished_at"] = int(time.time())
    quiz_store.update_quiz(quiz)
    return jsonify({"analytics": analytics, "enrolled_to_review": enrolled})


@bp.route("/api/quiz/<quiz_id>/analytics", methods=["GET"])
def api_quiz_analytics(quiz_id):
    """Return the analytics for a finished attempt (latest by default)."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    attempt_id = request.args.get("attempt_id")
    attempts = [a for a in (quiz.get("attempts") or [])
                if a.get("finished") and a.get("analytics")]
    if not attempts:
        return jsonify({"analytics": None})
    if attempt_id:
        hit = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
        if hit:
            return jsonify({"analytics": hit["analytics"]})
        return jsonify({"analytics": None})
    last = sorted(attempts, key=lambda a: a.get("finished_at", 0)).pop()
    return jsonify({"analytics": last["analytics"]})


# ─── FSRS review loop (SESSION 5) ────────────────────────────────────────────
#
# Missed questions are enrolled into a single shared review store
# (quizzes/review.json, keyed by quiz_id:qid) at attempt finish so the
# due-today queue spans quizzes. The Review pane pulls cards from
# GET /api/review/queue, applies an FSRS rating via
# POST /api/review/<quiz_id>/<qid>, and shows counts + strength-over-time via
# GET /api/review/stats. Scheduling is computed by py-fsrs with UTC now.

@bp.route("/api/review/queue", methods=["GET"])
def api_review_queue():
    """Due review cards across quizzes (question + provenance), oldest-due
    first, for one set."""
    set_name = request.args.get("set") or request.args.get("set_name") or "library"
    limit = cfg_int(load_cfg(), "quiz_review_limit", 50, 1, 500)
    cards = quiz_store.fetch_due_reviews(set_name=set_name, limit=limit)
    out = []
    for c in cards:
        question = c.get("question") or {}
        prov = question.get("provenance") or {}
        out.append({
            "key": c.get("key"),
            "quiz_id": c.get("quiz_id"),
            "qid": c.get("qid"),
            "set_name": c.get("set_name"),
            "type": c.get("type") or question.get("type") or "mcq",
            "question": question.get("q") or "",
            "choices": question.get("choices") or [],
            "answer": question.get("answer") or "",
            "section_title": prov.get("section_title") or "",
            "excerpt": (prov.get("excerpt_snippet") or "")[:400],
            "difficulty": question.get("difficulty") or "",
            "due": (c.get("fsrs") or {}).get("due"),
            "last_rating": c.get("last_rating"),
            "reviews": len(c.get("history") or []),
        })
    return jsonify({"cards": out, "count": len(out)})


@bp.route("/api/review/<quiz_id>/<qid>", methods=["POST"])
def api_review_rating(quiz_id, qid):
    """Apply an FSRS rating (again|hard|good|easy) to a stored review card."""
    if not load_cfg().get("fsrs_enabled", True):
        return jsonify({"error": "FSRS review is disabled in config."}), 400
    data = request.get_json(force=True, silent=True) or {}
    rating = data.get("rating") or request.form.get("rating")
    card = quiz_store.record_review_rating(quiz_id, qid, rating)
    if card is None:
        if rating not in ("again", "hard", "good", "easy"):
            return jsonify({"error": "Rating must be again|hard|good|easy."}), 400
        return jsonify({"error": "Review card not found (or FSRS unavailable)."}), 404
    history = card.get("history") or []
    last = history[-1] if history else {}
    return jsonify({
        "ok": True,
        "rating": rating,
        "due": (card.get("fsrs") or {}).get("due"),
        "stability": last.get("stability"),
        "last_rating": card.get("last_rating"),
    })


@bp.route("/api/review/stats", methods=["GET"])
def api_review_stats():
    """Review counts, retention rate, and a strength-over-time series."""
    set_name = request.args.get("set") or request.args.get("set_name") or "library"
    return jsonify(quiz_store.fetch_review_stats(set_name))


@bp.route("/api/review/summary", methods=["GET"])
def api_review_summary():
    """Per-quiz due counts (for the due badge on the generated-quiz list)."""
    set_name = request.args.get("set") or request.args.get("set_name") or "library"
    cards = quiz_store.fetch_due_reviews(set_name=set_name, limit=100000)
    counts = {}
    for c in cards:
        qid = c.get("quiz_id")
        counts[qid] = counts.get(qid, 0) + 1
    return jsonify({"due": counts})


@bp.route("/api/course", methods=["GET"])
def api_course():
    """Course/syllabus browser for one work: its unit list (with word counts)
    and a ``quizzed`` flag per unit derived from stored quizzes."""
    set_name = request.args.get("set") or request.args.get("set_name") or "library"
    work = (request.args.get("work") or "").strip()
    if not work:
        return jsonify({"error": "a work title is required"}), 400
    syllabus = quiz_syllabus.get_syllabus(work, set_name)
    units = syllabus.get("units") or []
    counts = _course_quizzed_counts(set_name, work)
    norm = lambda s: __import__("re").sub(
        r"[^a-z0-9]+", "", (s or "").strip().lower())
    out_units = []
    for u in units:
        quizzed = bool(counts.get(norm(u.get("title"))))
        out_units.append({
            "unit_id": u.get("unit_id"),
            "title": u.get("title"),
            "ordinal": u.get("ordinal"),
            "parent_ids": u.get("parent_ids") or [],
            "subtopics": u.get("subtopics") or [],
            "words": u.get("words") or 0,
            "quizzed": quizzed,
        })
    return jsonify({
        "title": work, "set_name": set_name,
        "units": out_units,
        "unit_count": len(out_units),
        "quizzed_count": sum(1 for u in out_units if u["quizzed"]),
        "warnings": syllabus.get("warnings") or [],
    })


@bp.route("/api/course/summary", methods=["POST"])
def api_course_summary():
    """Whole-work deep reading summary via catalog.map_reduce_summary."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "library"
    work = (data.get("work") or "").strip()
    if not work:
        return jsonify({"error": "a work title is required"}), 400
    try:
        collection = get_chat_collection(set_name)
        if collection.count() == 0:
            return jsonify({"error": "no collection available"}), 400
        import agent as _a
        matched = _a._match_titles(work, set_name, limit=4)
        if not matched:
            return jsonify({"error": f"no work matching {work!r}"}), 404
        summary = map_reduce_summary(
            set_name, matched, load_cfg(), collection, get_chat_embedder(),
            get_chat_client())
        return jsonify({"work": work, "set_name": set_name,
                        "matched": matched, "summary": summary})
    except (SystemExit, CollectionNotFound):
        return jsonify({"error": "collection unavailable"}), 400
    except Exception as e:
        print(f"[course] summary failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/api/course/unit-summary", methods=["POST"])
def api_course_unit_summary():
    """Short per-unit reading summary over that unit's parents via an LLM call."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "library"
    work = (data.get("work") or "").strip()
    unit_id = (data.get("unit_id") or "").strip()
    if not work or not unit_id:
        return jsonify({"error": "work and unit_id are required"}), 400
    syllabus = quiz_syllabus.get_syllabus(work, set_name)
    unit = next((u for u in (syllabus.get("units") or [])
                 if u.get("unit_id") == unit_id), None)
    if not unit:
        return jsonify({"error": "unit not found"}), 404
    parent_ids = unit.get("parent_ids") or []
    import agent as _a
    texts = _a._load_parent_texts(parent_ids, set_name) if parent_ids else {}
    parts = []
    for pid in parent_ids:
        p = texts.get(pid)
        if p:
            parts.append(" ".join((p.get("text") or "").split())[:900])
    if not parts:
        return jsonify({"error": "no parent text for this unit"}), 404
    cfg = load_cfg()
    prompt = (
        f"Write a concise reading summary (2-4 sentences) of the chapter/section "
        f"\"{unit.get('title')}\" of \"{work}\", covering its main points.\n\n"
        f"--- Section text ---\n\n" + "\n\n".join(parts))
    try:
        resp = get_chat_client().chat.completions.create(
            model=cfg.get("llm_model", "default"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=int(cfg.get("max_tokens_summary", 4096) or 4096),
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[course] unit summary failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    return jsonify({"work": work, "unit_id": unit_id,
                    "title": unit.get("title"), "summary": text})


@bp.route("/api/course/quiz-unit", methods=["POST"])
def api_course_quiz_unit():
    """Plan (and persist) a sections-scope spec for a single unit, so the UI can
    jump straight to generating a quiz on that unit."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "library"
    work = (data.get("work") or "").strip()
    unit_id = (data.get("unit_id") or "").strip()
    count = _as_int(data.get("count"), 10, 1, 100)
    if not work or not unit_id:
        return jsonify({"error": "work and unit_id are required"}), 400
    try:
        collection = get_chat_collection(set_name)
        plan = quiz_plan.plan_quiz(
            {"mode": "sections", "work": work, "unit_ids": [unit_id],
             "count": count}, set_name, load_cfg(),
            embedder=get_chat_embedder(), collection=collection)
    except (SystemExit, CollectionNotFound):
        return jsonify({"error": "collection unavailable"}), 400
    except Exception as e:
        print(f"[course] quiz-unit plan failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    if "error" in plan:
        return jsonify({"error": plan["error"]}), 400
    spec = plan["spec"]
    if not spec.get("id"):
        spec = quiz_store.create_spec(spec)
    return jsonify({"spec": spec})
