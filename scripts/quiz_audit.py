#!/usr/bin/env python3
"""quiz_audit.py - The 4-stage accuracy audit run on every generated quiz batch.

Generation is work-specific and deterministic: each batch of parents is sent to
the LLM, and every parsed question must clear the audit before it is kept.
The four stages are:

  0. Parse sanity   — the question is structurally valid (mcq answer letter
                      present in choices, >= 2 unique choices, no dup choices;
                      true_false answer in {true,false}; fill_blank has a
                      non-empty answer; short_answer has a non-empty answer).
  1. Grounding      — (no LLM) the question's ``excerpt_snippet`` must fuzzy-
                      match the cited parent's ACTUAL text (difflib ratio >=
                      ``quiz_grounding_ratio``), and its provenance (title /
                      section_title / parent_id) must resolve to a real parent
                      row whose title equals the cited title.
  2. Inline verify  — (LLM, gated by ``quiz_verify_pass``) given ONLY the
                      grounding parent text + question + choices + marked
                      answer, return a structured verdict (supported? yes/no).
                      Batched ~3 questions sharing a parent to amortize context.
  3. Dedupe/coverage— cross-batch: drop a question whose stemmed-token Jaccard
                      vs an earlier accepted question >= ``quiz_dedupe_jaccard``;
                      and report per-unit coverage for the build report.

The audit functions are pure (no heavy imports at module load) so they are easy
to unit-test and to reuse in a regression harness. Stage 2 needs the LLM client
plus the config; the others do not.
"""

import difflib
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Optional English stemmer, matching agent.stem_tokens exactly when available.
# snowballstemmer is a light dep; if missing (test sandbox) we fall back to
# plain lowercase tokens, which only weakens the dedupe threshold slightly.
try:
    from snowballstemmer import stemmer as _snowball
    _STEMMER = _snowball("english")
except Exception:  # pragma: no cover - sandbox without the dep
    _STEMMER = None


def _stem_tokens(text: str) -> list:
    """Lowercase + tokenize + English-stem (mirrors agent.stem_tokens) so the
    cross-batch dedupe threshold stays consistent with the rest of the app,
    WITHOUT importing agent (which pulls in chromadb/sentence_transformers)."""
    if not text:
        return []
    words = _TOKEN_RE.findall(text.lower())
    if _STEMMER is not None:
        return [_STEMMER.stemWord(w) for w in words]
    return words


def _norm(s) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _chip_excerpt(excerpt: str, parent_text: str) -> str:
    """Return the best fuzzy window of ``parent_text`` overlapping ``excerpt``.

    Because the model may paraphrase slightly, we slide over sentence-length
    windows of the parent text and return the one with the highest ratio;
    if nothing useful, return the (normalized) excerpt itself.
    """
    ex = _norm(excerpt)
    pt = _norm(parent_text)
    if not ex or not pt:
        return ex
    if ex in pt:
        return ex
    best = ("", 0.0)
    n = max(8, len(ex))
    m = len(pt)
    for i in range(0, max(1, m - n + 1), max(1, n // 2)):
        win = pt[i:i + n]
        r = difflib.SequenceMatcher(None, ex, win).ratio()
        if r > best[1]:
            best = (win, r)
    return best[0]


def _grounding_ratio(excerpt: str, parent_text: str) -> float:
    """Best difflib ratio between the excerpt and any window of the parent."""
    ex = _norm(excerpt)
    pt = _norm(parent_text)
    if not ex or not pt:
        return 0.0
    if ex in pt:
        return 1.0
    n = max(8, len(ex))
    m = len(pt)
    best = 0.0
    for i in range(0, max(1, m - n + 1), max(1, n // 2)):
        win = pt[i:i + n]
        r = difflib.SequenceMatcher(None, ex, win).ratio()
        if r > best:
            best = r
    return best


# ─── Stage 0: parse sanity (no LLM) ──────────────────────────────────────────

def parse_sanity(question: dict) -> tuple:
    """Return (ok, reason). Checks structural validity of a parsed question."""
    qtype = (question.get("type") or "mcq").strip().lower()
    q = (question.get("q") or "").strip()
    if not q:
        return False, "empty_question"
    if len(q) < 6:
        return False, "question_too_short"
    if qtype == "mcq":
        choices = question.get("choices") or []
        # choices are stored like "A) ...", "B) ..."
        if len(choices) < 2:
            return False, "too_few_choices"
        letters = [_letter_of(c) for c in choices]
        if len(set(letters)) != len(letters):
            return False, "duplicate_choice_letters"
        answer = (question.get("answer") or "").strip().upper()
        # answer may be "A", "A) ...", or the full text
        ans_letter = _letter_of(question.get("answer")) or answer
        if ans_letter not in letters:
            return False, "answer_missing_from_choices"
    elif qtype == "true_false":
        ans = (question.get("answer") or "").strip().lower()
        if ans not in ("true", "false"):
            return False, "invalid_true_false_answer"
    elif qtype in ("fill_blank", "short_answer"):
        if not (question.get("answer") or "").strip():
            return False, "empty_answer"
    return True, ""


def _letter_of(choice) -> str:
    """Extract the leading answer letter from 'A) ...' / 'A' / 'A. ...'."""
    s = (choice or "").strip().upper()
    m = re.match(r"^([A-D])\s*[).:]", s)
    if m:
        return m.group(1)
    m = re.match(r"^([A-D])$", s)
    return m.group(1) if m else ""


# ─── Stage 1: grounding (no LLM) ─────────────────────────────────────────────

def grounding_check(question: dict, parent: dict | None,
                    ground_ratio: float = 0.85) -> tuple:
    """Return (ok, reason). Validate the question's citation against the parent.

    ``parent`` is the resolved parents-table row ({parent_id, title,
    section_title, text}) for ``question.provenance.parent_id``.
    """
    prov = question.get("provenance") or {}
    if parent is None:
        return False, "parent_not_found"
    if not (parent.get("parent_id") or prov.get("parent_id")):
        return False, "missing_parent_id"
    # Title must match the actual parent's title (fuzzy, so truncation is fine).
    cited = (prov.get("title") or "").strip()
    real = (parent.get("title") or "").strip()
    if cited and real and not _titles_agree(cited, real):
        return False, "title_mismatch"
    excerpt = prov.get("excerpt_snippet") or ""
    if not excerpt:
        return False, "missing_excerpt"
    if not parent.get("text"):
        return False, "empty_parent_text"
    ratio = _grounding_ratio(excerpt, parent.get("text", ""))
    if ratio < ground_ratio:
        return False, f"weak_grounding:{ratio:.2f}"
    return True, ""


def _titles_agree(a: str, b: str) -> bool:
    an = _norm(a)
    bn = _norm(b)
    if not an or not bn:
        return True
    if an == bn or an in bn or bn in an:
        return True
    ad = {w for w in an.split() if len(w) > 1}
    bd = {w for w in bn.split() if len(w) > 1}
    return len(ad & bd) >= 2


# ─── Stage 3: cross-batch dedupe + coverage (no LLM) ─────────────────────────

def _jaccard(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_duplicate(candidate: dict, accepted_qs: list, threshold: float = 0.75) -> bool:
    """Return True if candidate's stemmed tokens overlap an accepted question."""
    cand = set(_stem_tokens(candidate.get("q") or ""))
    if not cand:
        return True
    for a in accepted_qs:
        acc = set(_stem_tokens(a.get("q") or ""))
        if acc and _jaccard(sorted(cand), sorted(acc)) >= threshold:
            return True
    return False


def coverage_map(accepted_qs: list, unit_allocations: dict) -> dict:
    """Return {unit_id: accepted_count} plus overall counts for the report."""
    per_unit = {}
    for q in accepted_qs:
        uid = (q.get("unit") or q.get("unit_id") or "")
        per_unit[uid] = per_unit.get(uid, 0) + 1
    total_req = 0
    for v in (unit_allocations or {}).values():
        try:
            total_req += int(v)
        except (TypeError, ValueError):
            total_req += 0
    return {
        "per_unit": per_unit,
        "total_accepted": len(accepted_qs),
        "total_requested": total_req,
    }


# ─── Deterministic audit API used by quiz_build ─────────────────────────────

def run_deterministic_audit(question: dict, parent: dict | None,
                            ground_ratio: float = 0.85) -> tuple:
    """Run the deterministic stages (0 parse sanity + 1 grounding).

    Returns (ok, reason). Stages 2 (LLM verify) and 3 (dedupe) are applied
    separately by the builder (they need a client / accepted-question state).
    """
    ok0, r0 = parse_sanity(question)
    if not ok0:
        return False, r0
    ok1, r1 = grounding_check(question, parent, ground_ratio=ground_ratio)
    if not ok1:
        return False, r1
    return True, "ok"
