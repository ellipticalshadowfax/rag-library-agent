"""Path-safety helpers for store ids (chat_store, quiz_store, _paths.safe_id)."""

import pytest

import _paths
import chat_store
import quiz_store

GOOD_ID = "0123456789ab"
TRAVERSAL_IDS = ["../x", "../../etc/passwd", "/etc/passwd", "a/b",
                 "..", ".", "abc", "0123456789AB", "0123456789ab/..",
                 "0123456789ab.json", "", None]


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_safe_id_rejects_non_generated_ids(bad):
    assert _paths.safe_id(bad) == ""


def test_safe_id_accepts_generated_shape():
    assert _paths.safe_id(GOOD_ID) == GOOD_ID


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_chat_path_maps_bad_ids_to_sentinel(bad):
    p = chat_store._path(bad)
    assert p == chat_store.CONV_DIR / "__invalid__.json"
    assert p.parent == chat_store.CONV_DIR


def test_chat_path_for_good_id(test_root):
    assert chat_store._path(GOOD_ID) == chat_store.CONV_DIR / f"{GOOD_ID}.json"


def test_chat_store_ignores_traversal_ids(test_root):
    assert chat_store.get_conversation("../../etc/passwd") is None
    assert chat_store.delete_conversation("../x") is False


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_quiz_paths_map_bad_ids_to_sentinel(bad):
    spec = quiz_store._spec_path(bad)
    quiz = quiz_store._quiz_path(bad)
    assert spec == quiz_store.QUIZ_DIR / "specs" / "__invalid__.json"
    assert quiz == quiz_store.QUIZ_DIR / "__invalid__.json"


def test_quiz_paths_for_good_id():
    assert quiz_store._spec_path(GOOD_ID).name == f"{GOOD_ID}.json"
    assert quiz_store._spec_path(GOOD_ID).parent == quiz_store.QUIZ_DIR / "specs"
    assert quiz_store._quiz_path(GOOD_ID) == quiz_store.QUIZ_DIR / f"{GOOD_ID}.json"


@pytest.mark.parametrize("value", ["../x", "a/b", "c\\d", "sp ace", "a\x00b"])
def test_safe_name_strips_separators(value):
    name = quiz_store._safe(value)
    assert "/" not in name and "\\" not in name and "\x00" not in name


@pytest.mark.parametrize("value", [".", "..", "...", ""])
def test_safe_name_rejects_dot_only_component(value):
    name = quiz_store._safe(value)
    assert name not in (".", "..")
    assert name.strip(".") != ""
