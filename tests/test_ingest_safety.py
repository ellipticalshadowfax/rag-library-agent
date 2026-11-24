"""ingest.py safety: no data loss on embed/upsert failure, embedder identity
guard + --rebuild, empty-file cleanup, single BM25 df rebuild, no torch setup
for remote backends. Fake embedder, tmp RAG_ROOT, real chromadb + sqlite."""

import hashlib
import sqlite3

import chromadb
import numpy as np
import pytest

import _paths
import ingest


class FakeEmbedder:
    def __init__(self, dim=4, encode_dim=None, fail_on=None):
        self.dim = dim
        self.encode_dim = encode_dim or dim
        self.fail_on = fail_on  # callable(batch) -> raise
        self.calls = 0

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, batch, **kw):
        self.calls += 1
        if self.fail_on:
            self.fail_on(batch)
        out = []
        for t in batch:
            h = hashlib.sha256(t.encode()).digest()
            out.append([(h[i % 32] + 1) / 256 for i in range(self.encode_dim)])
        return np.array(out, dtype="float32")


def _cfg(**over):
    cfg = {"embed_backend": "ollama", "embed_model": "fake-a", "embed_dim": 4,
           "chunking_strategy": "parent_child", "chunk_tokens": 20,
           "chunk_overlap": 5, "ingest_batch_size": 4, "embed_batch_size": 4}
    cfg.update(over)
    return cfg


def _text(seed, n=150):
    return " ".join(f"{seed}word{i % 17} alpha{i}." for i in range(n))


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setenv("RAG_ROOT", str(root))
    state = {"emb": FakeEmbedder()}
    monkeypatch.setattr(_paths, "build_embedder", lambda cfg, *a, **k: state["emb"])
    return root, lib, state


def _write(lib, name, text):
    p = lib / name
    p.write_text(text, encoding="utf-8")
    return p


def _col(root):
    return chromadb.PersistentClient(path=str(root / "index")).get_collection("sset")


def _db(root):
    return sqlite3.connect(str(root / "manifest.db"))


def _bm25_ids(root):
    return {r[0] for r in _db(root).execute(
        "SELECT DISTINCT doc_id FROM bm25_tokens WHERE set_name='sset'")}


def _run(lib, cfg=None, **kw):
    ingest.ingest(str(lib), "sset", cfg or _cfg(), **kw)


# ── 8a: embed/upsert failure keeps the old chunks ───────────────────────────
def test_embed_failure_keeps_old_chunks(env):
    root, lib, state = env
    p = _write(lib, "a.txt", _text("one"))
    _run(lib)
    before = _col(root).get(include=["documents"])
    ids_before, bm_before = set(before["ids"]), _bm25_ids(root)
    parents_before = _db(root).execute("SELECT COUNT(*) FROM parents").fetchone()[0]
    assert ids_before and bm_before and parents_before

    p.write_text(_text("two", 200), encoding="utf-8")

    def boom(batch):
        raise RuntimeError("ollama down")
    state["emb"] = FakeEmbedder(fail_on=boom)
    _run(lib)

    after = _col(root).get(include=["documents"])
    assert set(after["ids"]) == ids_before
    assert sorted(after["documents"]) == sorted(before["documents"])
    assert _bm25_ids(root) == bm_before
    assert _db(root).execute("SELECT COUNT(*) FROM parents").fetchone()[0] == parents_before


def test_late_batch_embed_failure_does_not_delete(env):
    root, lib, state = env
    p = _write(lib, "a.txt", _text("one"))
    _run(lib)
    before = set(_col(root).get()["ids"])
    p.write_text(_text("two", 300), encoding="utf-8")
    n = {"i": 0}

    def fail_second(batch):
        n["i"] += 1
        if n["i"] >= 2:
            raise RuntimeError("late failure")
    state["emb"] = FakeEmbedder(fail_on=fail_second)
    _run(lib)
    got = _col(root).get(include=["documents"])
    assert set(got["ids"]) == before
    assert not any("two" in d for d in got["documents"])


def test_wrong_dim_upsert_failure_keeps_old_chunks(env):
    root, lib, state = env
    p = _write(lib, "a.txt", _text("one"))
    _run(lib)
    before = _col(root).get(include=["documents"])
    p.write_text(_text("two"), encoding="utf-8")
    state["emb"] = FakeEmbedder(dim=4, encode_dim=7)  # reports 4, returns 7
    _run(lib)
    after = _col(root).get(include=["documents"])
    assert sorted(after["documents"]) == sorted(before["documents"])
    assert _bm25_ids(root) == set(before["ids"])


def test_shrinking_file_removes_stale_tail(env):
    root, lib, state = env
    p = _write(lib, "a.txt", _text("one", 300))
    _run(lib)
    n_before = _col(root).count()
    p.write_text(_text("one", 60), encoding="utf-8")
    _run(lib)
    col = _col(root)
    assert 0 < col.count() < n_before
    assert _bm25_ids(root) == set(col.get()["ids"])


# ── 8b: embedder identity ───────────────────────────────────────────────────
def test_identity_stored_and_cosine_kept(env):
    root, lib, state = env
    _write(lib, "a.txt", _text("one"))
    _run(lib, _cfg(embed_passage_prefix="passage: "))
    col = _col(root)
    md = col.metadata
    assert md["rag_embed_backend"] == "ollama"
    assert md["rag_embed_model"] == "fake-a"
    assert md["rag_embed_dim"] == 4
    assert md["rag_embed_passage_prefix"] == "passage: "
    assert "rag_embed_query_prefix" in md
    assert col.configuration["hnsw"]["space"] == "cosine"


@pytest.mark.parametrize("over,field", [
    ({"embed_model": "fake-b"}, "model"),
    ({"embed_backend": "local"}, "backend"),
    ({"embed_passage_prefix": "doc: "}, "passage_prefix"),
])
def test_identity_mismatch_refuses_before_touching(env, capsys, over, field):
    root, lib, state = env
    p = _write(lib, "a.txt", _text("one"))
    _run(lib)
    before = _col(root).get(include=["documents"])
    bm = _bm25_ids(root)
    p.write_text(_text("two"), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        _run(lib, _cfg(**over))
    assert ei.value.code not in (0, None)
    out = capsys.readouterr().out
    assert field in out and "--rebuild" in out
    after = _col(root).get(include=["documents"])
    assert sorted(after["documents"]) == sorted(before["documents"])
    assert _bm25_ids(root) == bm


def test_dim_mismatch_refuses(env):
    root, lib, state = env
    _write(lib, "a.txt", _text("one"))
    _run(lib)
    state["emb"] = FakeEmbedder(dim=6)
    with pytest.raises(SystemExit):
        _run(lib, _cfg(embed_dim=6), force=True)


def test_legacy_collection_adopted_with_warning(env, capsys):
    root, lib, state = env
    _write(lib, "a.txt", _text("one"))
    _run(lib)
    _col(root).modify(metadata={"legacy": "yes"})  # no identity (pre-fix index)
    capsys.readouterr()
    _run(lib, _cfg(embed_model="whatever"), force=True)
    md = _col(root).metadata
    assert md["rag_embed_model"] == "whatever"
    assert "unknown" in capsys.readouterr().out.lower()


def test_rebuild_switches_embedder(env):
    root, lib, state = env
    _write(lib, "a.txt", _text("one"))
    _write(lib, "b.txt", _text("two"))
    _run(lib)
    state["emb"] = FakeEmbedder(dim=6)
    _run(lib, _cfg(embed_model="fake-b", embed_dim=6), rebuild=True)
    col = _col(root)
    assert col.metadata["rag_embed_model"] == "fake-b"
    assert col.metadata["rag_embed_dim"] == 6
    assert col.count() > 0
    assert _bm25_ids(root) == set(col.get()["ids"])
    assert _db(root).execute("SELECT COUNT(*) FROM files WHERE set_name='sset'").fetchone()[0] == 2


def test_rebuild_cli_flag_exists():
    import argparse
    import inspect
    assert "--rebuild" in inspect.getsource(ingest.main)
    assert "rebuild" in inspect.signature(ingest.ingest).parameters
    assert argparse  # silence linters


# ── 14: BM25 df ─────────────────────────────────────────────────────────────
def _df_consistent(root):
    db = _db(root)
    want = dict(((t, c) for t, c in db.execute(
        "SELECT token, COUNT(DISTINCT doc_id) FROM bm25_tokens "
        "WHERE set_name='sset' GROUP BY token")))
    got = dict(((t, c) for t, c in db.execute(
        "SELECT token, doc_freq FROM bm25_df WHERE set_name='sset'")))
    return want == got and bool(want)


def test_df_rebuilt_once_per_ingest(env, monkeypatch):
    root, lib, state = env
    for i in range(4):
        _write(lib, f"f{i}.txt", _text(f"s{i}"))
    calls = []
    orig = ingest.Manifest.rebuild_bm25_df
    monkeypatch.setattr(ingest.Manifest, "rebuild_bm25_df",
                        lambda self, set_name="": (calls.append(set_name), orig(self, set_name))[1])
    _run(lib)
    assert len(calls) == 1
    assert _df_consistent(root)


def test_df_consistent_after_mid_run_stop(env):
    root, lib, state = env
    for i in range(3):
        _write(lib, f"f{i}.txt", _text(f"s{i}"))
    n = {"i": 0}

    def stop_later(batch):
        n["i"] += 1
        if n["i"] >= 6:  # a later file, after at least one fully written
            raise SystemExit(0)  # what _handle_stop raises
    state["emb"] = FakeEmbedder(fail_on=stop_later)
    with pytest.raises(SystemExit):
        _run(lib)
    assert _bm25_ids(root)
    assert _df_consistent(root)


# ── 15: empty files ─────────────────────────────────────────────────────────
def test_blank_file_purges_old_index_and_updates_manifest(env):
    root, lib, state = env
    p = _write(lib, "a.txt", _text("one"))
    _run(lib)
    assert _col(root).count() and _bm25_ids(root)
    p.write_text("   \n\n", encoding="utf-8")
    _run(lib)
    assert _col(root).count() == 0
    assert _bm25_ids(root) == set()
    assert _db(root).execute("SELECT COUNT(*) FROM parents").fetchone()[0] == 0
    m = ingest.Manifest(root / "manifest.db")
    st = p.stat()
    assert m.is_current("a.txt", st.st_mtime, st.st_size, "sset", fpath=p)


def test_no_chunks_purges_and_upserts_manifest(env, monkeypatch):
    root, lib, state = env
    p = _write(lib, "a.txt", _text("one"))
    _run(lib)
    p.write_text(_text("two"), encoding="utf-8")
    monkeypatch.setattr(ingest, "chunk_text_recursive", lambda *a, **k: [])
    _run(lib)
    assert _col(root).count() == 0
    assert _bm25_ids(root) == set()
    m = ingest.Manifest(root / "manifest.db")
    st = p.stat()
    assert m.is_current("a.txt", st.st_mtime, st.st_size, "sset", fpath=p)


# ── 17: no torch setup for remote backends ──────────────────────────────────
def test_remote_backend_skips_torch_setup(env, monkeypatch):
    root, lib, state = env
    _write(lib, "a.txt", _text("one"))

    def nope(*a, **k):
        raise AssertionError("torch/device setup must not run for remote backend")
    monkeypatch.setattr(ingest.torch, "set_num_threads", nope)
    monkeypatch.setattr(ingest.torch, "get_num_threads", nope)
    monkeypatch.setattr(ingest, "resolve_device", nope)
    _run(lib)
    assert _col(root).count() > 0


def test_local_backend_still_sets_threads(env, monkeypatch):
    root, lib, state = env
    _write(lib, "a.txt", _text("one"))
    seen = []
    monkeypatch.setattr(ingest.torch, "set_num_threads", lambda n: seen.append(n))
    _run(lib, _cfg(embed_backend="local"))
    assert seen
