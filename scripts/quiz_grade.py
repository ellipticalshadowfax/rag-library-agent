#!/usr/bin/env python3
"""quiz_grade.py - SESSION 4 grading and analytics for the Study-tab player.

Grading is deterministic for mcq / true_false / fill_blank and LLM-assisted for
short_answer (using only the question's own provenance excerpt — small, grounded
calls, no context overload), gated by ``quiz_grade_llm``. Analytics aggregate
accuracy per unit / difficulty / type so strengths and weaknesses, plus
provenance-linked study points, are visible at the end of an attempt.

A graded response is:
    {
        "correct": bool,
        "expected": str,   # the key/canonical answer (for review)
        "feedback": str,   # human-readable feedback (LLM feedback for short)
        "score": float,    # 0.0 or 1.0 for now (short_answer may refine later)
    }
"""

import re

import study


def _norm(s):
    """Same tolerant normalization used by study.title_matches."""
    return study.norm_tokens(s)


def _as_bool(v) -> bool | None:
    """Coerce a true_false response (True/False/yes/no/1/0) to a bool."""
    s = _norm(v)
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return None


def _answer_letter(answer) -> str:
    """Extract the letter (A..D) from an mcq answer value (e.g. 'A' or 'B) ...')."""
    m = re.match(r"^\s*([A-D])\s*[).:]*", str(answer or ""))
    return m.group(1).upper() if m else ""


def _choice_letter(choice) -> str:
    """Extract the leading letter from a rendered choice (e.g. 'B) ...')."""
    m = re.match(r"^\s*([A-D])\s*[).:]", str(choice or ""))
    return m.group(1).upper() if m else ""


def _grade_mcq(question, response):
    expected = _answer_letter(question.get("answer") or "")
    given = _answer_letter(response) or _choice_letter(response)
    if not expected or not given:
        return None
    return {
        "correct": given == expected,
        "expected": expected or (question.get("answer") or ""),
        "feedback": "",
        "score": 1.0 if given == expected else 0.0,
    }


def _grade_true_false(question, response):
    expected = _as_bool(question.get("answer"))
    given = _as_bool(response)
    if expected is None or given is None:
        return None
    return {
        "correct": given == expected,
        "expected": "true" if expected else "false",
        "feedback": "",
        "score": 1.0 if given == expected else 0.0,
    }


def _grade_fill_blank(question, response):
    """Normalized token match against the expected text."""
    expected = _norm(question.get("answer") or "")
    given = _norm(response)
    if not expected:
        return None
    correct = expected == given or (bool(given) and expected in given)
    return {
        "correct": correct,
        "expected": question.get("answer") or "",
        "feedback": "",
        "score": 1.0 if correct else 0.0,
    }


def _grade_short_llm(question, response, cfg, client):
    """LLM-assisted grading: structured verdict {score 0..1, feedback} given the
    question, its provenance excerpt, and the response — nothing else."""
    prov = question.get("provenance") or {}
    excerpt = prov.get("excerpt_snippet") or ""
    prompt = (
        "You are grading a practice-quiz short answer. Use ONLY the provided "
        "source excerpt to judge whether the user's response is a correct "
        "answer to the question. Return a JSON object:\n"
        "{'score': <number 0..1>, 'feedback': '<short 1-line comment>'}\n\n"
        f"QUESTION: {question.get('q')}\n\n"
        f"SOURCE EXCERPT: {excerpt}\n\n"
        f"EXPECTED ANSWER: {question.get('answer')}\n\n"
        f"USER RESPONSE: {response or '(blank)'}\n\n"
        "BE FAIR: accept paraphrases that capture the key idea from the "
        "excerpt. score = 1 for fully correct, 0.5 for partially correct, 0 "
        "for wrong.")
    try:
        resp = client.chat.completions.create(
            model=cfg.get("llm_model", "default"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Tolerant of markdown fences / leading prose.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON")
        import json
        data = json.loads(text[start:end + 1])
        score = float(data.get("score", 0.0))
        feedback = str(data.get("feedback", "") or "")
        score = max(0.0, min(1.0, score))
    except Exception:
        # Degrade to a deterministic token overlap when the LLM path fails.
        return _grade_short_tokens(question, response)
    return {
        "correct": score >= 0.5,
        "expected": question.get("answer") or "",
        "feedback": feedback,
        "score": score,
    }


def _grade_short_tokens(question, response):
    """Deterministic fallback for short answers: significant-token overlap."""
    exp_tokens = {w for w in _norm(question.get("answer") or "").split() if len(w) > 2}
    given = _norm(response)
    if not exp_tokens or not given:
        given_tokens = set()
    else:
        given_tokens = {w for w in given.split() if len(w) > 2}
    if not exp_tokens:
        return None
    overlap = len(exp_tokens & given_tokens) / len(exp_tokens)
    correct = bool(given) and overlap >= 0.6
    return {
        "correct": correct,
        "expected": question.get("answer") or "",
        "feedback": "",
        "score": 1.0 if correct else 0.0,
    }


def grade_answer(question, response, cfg, client=None):
    """Grade one answer. Returns a dict {correct, expected, feedback, score} or
    None when the question/response could not be graded deterministically.

    ``client`` is only used for LLM-assisted short-answer grading (gated by
    ``quiz_grade_llm``); it may be None otherwise.
    """
    qtype = (question or {}).get("type", "mcq").strip().lower()
    if qtype == "mcq":
        return _grade_mcq(question, response)
    if qtype == "true_false":
        return _grade_true_false(question, response)
    if qtype == "fill_blank":
        return _grade_fill_blank(question, response)
    if qtype == "short_answer":
        if cfg.get("quiz_grade_llm", True) and client is not None:
            return _grade_short_llm(question, response, cfg, client)
        return _grade_short_tokens(question, response)
    return None


# ─── Analytics ────────────────────────────────────────────────────────────────

def _accuracy(correct, total):
    return round(correct / total, 4) if total else 0.0


def compute_analytics(attempt, questions, weak_threshold: float = 0.6):
    """Aggregate per-unit / per-difficulty / per-type accuracy and study points
    from a list of graded question results.

    ``attempt`` is a dict mapping question qid -> graded result dict (from
    grade_answer). ``questions`` is the quiz's full question list. ``weak_
    threshold`` is the per-unit accuracy below which a unit is flagged as weak
    (from the ``quiz_weak_threshold`` config key). Returns an analytics dict:
        {score, answered, total, correct, accuracy, per_unit{unit:{correct,
         total, accuracy}}, per_difficulty{}, per_type{}, weak_units[], count,
         study_points[{section_title, unit, excerpt}]}
    """
    by_qid = {qid: q for qid, q in enumerate_all(questions)}
    total = len(by_qid)
    per_unit = {}
    per_diff = {}
    per_type = {}
    study_points = []
    correct_total = 0
    answered = 0

    def bucket(acc, key, correct):
        b = acc.setdefault(key, {"correct": 0, "total": 0})
        b["total"] += 1
        if correct:
            b["correct"] += 1

    for qid, q in by_qid.items():
        graded = (attempt or {}).get(str(qid))
        if graded is None:
            continue
        answered += 1
        ok = bool(graded.get("correct"))
        if ok:
            correct_total += 1
        unit = q.get("unit") or (q.get("provenance") or {}).get("section_title") or "?"
        bucket(per_unit, unit, ok)
        bucket(per_diff, q.get("difficulty", "?"), ok)
        bucket(per_type, q.get("type", "?"), ok)
        if not ok:
            prov = q.get("provenance") or {}
            study_points.append({
                "qid": qid,
                "question": q.get("q") or "",
                "section_title": prov.get("section_title") or unit,
                "unit": unit,
                "excerpt": (prov.get("excerpt_snippet") or "")[:400],
                "given": (graded or {}).get("expected", ""),
            })

    weak_units = [
        u for u, b in per_unit.items()
        if b["total"] > 0 and _accuracy(b["correct"], b["total"]) < weak_threshold
    ]

    return {
        "answered": answered,
        "total": total,
        "correct": correct_total,
        "accuracy": _accuracy(correct_total, answered) if answered else 0.0,
        "per_unit": per_unit,
        "per_difficulty": per_diff,
        "per_type": per_type,
        "weak_units": weak_units,
        "study_points": study_points,
    }


def enumerate_all(questions):
    """Yield (qid, question) for the quiz's question list.

    qid is the question's own ``qid`` field (e.g. 'abc-0001') if present, else
    the array index. Grading results are keyed by this identifier so analytics
    can line the graded answers back up to the questions.
    """
    for i, q in enumerate(questions or []):
        yield (q.get("qid") or str(i)), q
