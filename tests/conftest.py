"""Shared pytest setup.

RAG_ROOT must point at a throwaway directory BEFORE any module under scripts/
is imported, because several modules (server, chat_store, quiz_store) compute
their data paths at import time. The real config.json, config.local.json,
index/, manifest.db, conversations/ and quizzes/ are never touched.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# Force (not setdefault): a developer's shell may export a real RAG_ROOT.
TEST_ROOT = Path(tempfile.mkdtemp(prefix="rag-tests-"))
os.environ["RAG_ROOT"] = str(TEST_ROOT)
for _var in ("RAG_ALLOWED_HOSTS", "RAG_HOST"):
    os.environ.pop(_var, None)

assert TEST_ROOT.resolve() != REPO.resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_DATA_ENTRIES = ("config.json", "config.local.json", "conversations", "quizzes")


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


@pytest.fixture
def test_root():
    """Reset the throwaway data root to a minimal config before each test."""
    for name in _DATA_ENTRIES:
        p = TEST_ROOT / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    (TEST_ROOT / "config.json").write_text(
        '{"llm_model": "test-model"}\n', encoding="utf-8")
    return TEST_ROOT


@pytest.fixture
def server(test_root):
    import server as srv  # noqa: WPS433 (import after RAG_ROOT is set)
    srv.app.testing = True
    return srv


@pytest.fixture
def client(server):
    return server.app.test_client()
