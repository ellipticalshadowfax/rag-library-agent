"""_as_int clamping (scripts/server.py) and deterministic quiz grading
(scripts/quiz_grade.py). No LLM, GPU or index is used: the LLM client is a fake
that either raises or returns a fixed string."""

import pytest

import quiz_grade


# ─── _as_int ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    ("7", 7), (7, 7), (2.9, 2), (True, 1), ("-3", -3),
])
def test_as_int_coerces_numbers(server, value, expected):
    assert server._as_int(value, 4) == expected


@pytest.mark.parametrize("value", [None, "abc", "2.9", [], {}])
def test_as_int_wrong_type_falls_back_to_default(server, value):
    assert server._as_int(value, 4) == 4


def test_as_int_clamps_high_and_low(server):
    assert server._as_int(10, 1, lo=1, hi=8) == 8
    assert server._as_int(-5, 1, lo=0) == 0
    assert server._as_int(3, 1, lo=1, hi=8) == 3


def test_as_int_wrong_type_and_out_of_range_default(server):
    # Combination: junk input AND the default itself lies outside [lo, hi].
    assert server._as_int("junk", 50, lo=1, hi=8) == 8
    assert server._as_int([], 0, lo=1, hi=8) == 1


# ─── quiz_grade: deterministic paths ─────────────────────────────────────────

class _ExplodingClient:
    """Fails the test if any LLM call is attempted."""

    class chat:  # noqa: N801 (mirrors the OpenAI client shape)
        class completions:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                raise AssertionError("LLM must not be called here")


class _RaisingClient:
    class chat:  # noqa: N801
        class completions:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("backend down")


class _FixedClient:
    def __init__(self, text):
        self._text = text

    @property
    def chat(self):
        text = self._text

        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            @staticmethod
            def create(**kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        return _Chat()


CFG = {"quiz_grade_llm": True}


def test_mcq_accepts_letter_or_choice_text():
    q = {"type": "mcq", "answer": "B) Paris"}
    assert quiz_grade.grade_answer(q, "B", CFG)["correct"] is True
    assert quiz_grade.grade_answer(q, "B) Paris", CFG)["correct"] is True
    # Observed behaviour, pinned as-is: letter matching is case-sensitive, so a
    # lowercase "b" is ungradable (None), not graded wrong.
    assert quiz_grade.grade_answer(q, "b", CFG) is None
    wrong = quiz_grade.grade_answer(q, "A", CFG)
    assert wrong == {"correct": False, "expected": "B", "feedback": "",
                     "score": 0.0}


def test_mcq_blank_or_unparseable_is_ungradable():
    q = {"type": "mcq", "answer": "B"}
    assert quiz_grade.grade_answer(q, "", CFG) is None
    assert quiz_grade.grade_answer(q, "zebra", CFG) is None
    assert quiz_grade.grade_answer({"type": "mcq", "answer": ""}, "A", CFG) is None


@pytest.mark.parametrize("response, correct", [
    ("yes", True), ("TRUE", True), ("1", True), ("no", False), ("0", False),
])
def test_true_false_grading(response, correct):
    res = quiz_grade.grade_answer({"type": "true_false", "answer": "true"},
                                  response, CFG)
    assert res["correct"] is correct
    assert res["score"] == (1.0 if correct else 0.0)


def test_true_false_unparseable_is_ungradable():
    q = {"type": "true_false", "answer": "true"}
    assert quiz_grade.grade_answer(q, "maybe", CFG) is None


def test_fill_blank_normalises_punctuation_and_case():
    q = {"type": "fill_blank", "answer": "Photosynthesis"}
    assert quiz_grade.grade_answer(q, "photosynthesis.", CFG)["correct"] is True
    res = quiz_grade.grade_answer(q, "", CFG)
    assert res["correct"] is False and res["score"] == 0.0


def test_short_answer_without_llm_uses_token_overlap():
    q = {"type": "short_answer",
         "answer": "The mitochondria produces energy for cells"}
    # 4 of 6 significant tokens = 0.67 >= 0.6
    assert quiz_grade.grade_answer(
        q, "mitochondria produces energy cells", {"quiz_grade_llm": True})["correct"]
    # 1 of 6 = 0.17
    assert quiz_grade.grade_answer(
        q, "energy", {"quiz_grade_llm": True})["correct"] is False


def test_short_answer_llm_disabled_never_calls_client():
    q = {"type": "short_answer", "answer": "mitochondria energy cells"}
    res = quiz_grade.grade_answer(q, "mitochondria energy cells",
                                  {"quiz_grade_llm": False}, _ExplodingClient())
    assert res["correct"] is True


def test_short_answer_llm_failure_degrades_to_tokens():
    q = {"type": "short_answer", "answer": "mitochondria energy cells"}
    res = quiz_grade.grade_answer(q, "mitochondria energy cells", CFG,
                                  _RaisingClient())
    assert res["correct"] is True and res["score"] == 1.0


def test_short_answer_llm_threshold_and_fence_parsing():
    q = {"type": "short_answer", "answer": "x"}
    text = 'Sure:\n```json\n{"score": 0.5, "feedback": "partly"}\n```'
    res = quiz_grade.grade_answer(q, "x", CFG, _FixedClient(text))
    assert res == {"correct": True, "expected": "x", "feedback": "partly",
                   "score": 0.5}


def test_short_answer_llm_wrong_type_and_out_of_range_combined():
    # Combination case: score is a numeric string outside 0..1 (9), and
    # feedback is a non-string (3). Expect clamp to 1.0 and str() coercion.
    q = {"type": "short_answer", "answer": "x"}
    text = '{"score": "9", "feedback": 3}'
    res = quiz_grade.grade_answer(q, "x", CFG, _FixedClient(text))
    assert res["score"] == 1.0
    assert res["correct"] is True
    assert res["feedback"] == "3"


def test_short_answer_llm_non_numeric_score_falls_back_to_tokens():
    q = {"type": "short_answer", "answer": "mitochondria energy cells"}
    text = '{"score": "very good", "feedback": "nice"}'
    res = quiz_grade.grade_answer(q, "mitochondria energy cells", CFG,
                                  _FixedClient(text))
    assert res["correct"] is True and res["feedback"] == ""


def test_unknown_type_is_ungradable():
    assert quiz_grade.grade_answer({"type": "essay", "answer": "x"}, "x", CFG) is None


def test_non_string_response_is_ungradable_not_a_crash():
    # Combination case: wrong-typed response (int) on an mcq item.
    q = {"type": "mcq", "answer": "A"}
    assert quiz_grade.grade_answer(q, 2, CFG) is None
    assert quiz_grade.grade_answer(q, None, CFG) is None
