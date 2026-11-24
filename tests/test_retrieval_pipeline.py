"""Retrieval-pipeline regressions (scripts/agent.py).

Covers the hybrid retrieval path: Chroma ``where`` clause composition, the
BM25 leg's metadata filtering / stale-posting handling, RRF fusion of
overlapping legs, config-driven top-k cap, title-mode fusion+rerank, and the
empty-collection BM25 index.

No LLM, GPU or real index is used: embeddings are hand-made vectors and the
embedder is a fake object, so everything runs against in-memory Chroma
collections.
"""

import numpy as np
import pytest
import chromadb
from rank_bm25 import BM25Okapi

import agent


class FakeEmbedder:
    """Deterministic embedder: every text maps to the same unit-ish vector."""

    def encode(self, texts, prompt_name=None):
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype="float32")


def _collection(name, rows):
    """Build a cosine in-memory collection from (id, text, embedding, meta)."""
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    if not rows:
        return col
    col.add(
        ids=[r[0] for r in rows],
        documents=[r[1] for r in rows],
        embeddings=[r[2] for r in rows],
        metadatas=[r[3] for r in rows],
    )
    return col


def _hit(cid, document="", metadata=None, distance=None, bm25_score=None):
    h = {
        "id": cid,
        "document": document,
        "metadata": metadata or {},
        "distance": distance,
    }
    if bm25_score is not None:
        h["bm25_score"] = bm25_score
    return h


def _stub_bm25(name, docs):
    """Cache a real BM25 index over {id: text} without touching Chroma/manifest."""
    ids = list(docs)
    tokdocs = [agent.tokenize(docs[i]) for i in ids]
    agent._bm25_cache[name] = {
        "bm25": BM25Okapi(tokdocs), "ids": ids, "df": {},
    }


@pytest.fixture(autouse=True)
def _clear_bm25_cache():
    agent._bm25_cache.clear()
    yield
    agent._bm25_cache.clear()


@pytest.fixture
def embedder():
    return FakeEmbedder()


# ─── 1. Chroma where-clause composition ──────────────────────────────────────

def test_retrieve_combines_kind_and_title_filters(embedder):
    col = _collection("tt_where_and", [
        ("f_a", "fiction alpha", [1.0, 0.0, 0.0],
         {"kind": "fiction", "title": "A"}),
        ("f_b", "fiction beta", [1.0, 0.0, 0.0],
         {"kind": "fiction", "title": "B"}),
        ("n_a", "nonfiction alpha", [1.0, 0.0, 0.0],
         {"kind": "nonfiction", "title": "A"}),
    ])
    hits = agent.retrieve(
        "alpha", embedder, col, top_k=10, filter_kind="fiction",
        where_extra={"title": {"$in": ["A"]}},
    )
    assert [h["id"] for h in hits] == ["f_a"]


def test_retrieve_single_filter_is_unwrapped_and_works(embedder):
    col = _collection("tt_where_one", [
        ("f_a", "fiction alpha", [1.0, 0.0, 0.0],
         {"kind": "fiction", "title": "A"}),
        ("n_a", "nonfiction alpha", [1.0, 0.0, 0.0],
         {"kind": "nonfiction", "title": "A"}),
    ])
    hits = agent.retrieve("alpha", embedder, col, top_k=10,
                          filter_kind="nonfiction")
    assert [h["id"] for h in hits] == ["n_a"]


# ─── 2. BM25 leg honours metadata filters ────────────────────────────────────

@pytest.fixture
def kind_collection():
    return _collection("tt_kind", [
        ("f1", "dragon dragon sword", [1.0, 0.0, 0.0],
         {"kind": "fiction", "title": "Dragon Book"}),
        ("n1", "dragon garden", [1.0, 0.0, 0.0],
         {"kind": "nonfiction", "title": "Garden Book"}),
    ])


def test_bm25_leg_respects_filter_kind(kind_collection):
    _stub_bm25("tt_kind", {
        "f1": "dragon dragon sword",
        "n1": "dragon garden",
    })
    fiction = agent._bm25_leg("tt_kind", "dragon", set(), kind_collection,
                              filter_kind="fiction")
    nonfiction = agent._bm25_leg("tt_kind", "dragon", set(), kind_collection,
                                 filter_kind="nonfiction")
    assert [h["id"] for h in fiction] == ["f1"]
    assert [h["id"] for h in nonfiction] == ["n1"]


def test_bm25_leg_respects_where_extra(kind_collection):
    _stub_bm25("tt_kind", {
        "f1": "dragon dragon sword",
        "n1": "dragon garden",
    })
    hits = agent._bm25_leg(
        "tt_kind", "dragon", set(), kind_collection,
        filter_kind="nonfiction",
        where_extra={"title": {"$in": ["Garden Book", "Dragon Book"]}},
    )
    assert [h["id"] for h in hits] == ["n1"]


# ─── 3. RRF rewards agreement between legs ───────────────────────────────────

def test_overlapping_top_hit_wins_fusion(embedder, monkeypatch):
    rows = [
        ("shared", "zebra zebra zebra", [1.0, 0.0, 0.0],
         {"kind": "nonfiction", "title": "Shared"}),
        ("x", "apple", [1.0, 0.0, 0.0], {"kind": "nonfiction", "title": "X"}),
        ("y", "banana", [1.0, 0.0, 0.0], {"kind": "nonfiction", "title": "Y"}),
        ("b1", "zebra", [1.0, 0.0, 0.0], {"kind": "nonfiction", "title": "B1"}),
    ]
    # Dense pool deliberately ranks "shared" second, BM25 ranks it first.
    # Filler docs keep "zebra" rare enough for BM25 to assign it positive idf.
    filler = [("f%d" % i, w, [1.0, 0.0, 0.0],
               {"kind": "nonfiction", "title": "F%d" % i})
              for i, w in enumerate(["cherry", "durian", "elder", "fig",
                                     "grape", "honey", "iris"])]
    col = _collection("tt_fuse", rows + filler)
    _stub_bm25("tt_fuse", {r[0]: r[1] for r in rows + filler})
    dense = [
        _hit("x", "apple", {"kind": "nonfiction", "title": "X"}, distance=0.1),
        _hit("shared", "zebra zebra zebra",
             {"kind": "nonfiction", "title": "Shared"}, distance=0.3),
        _hit("y", "banana", {"kind": "nonfiction", "title": "Y"}, distance=0.5),
    ]

    def fake_retrieve(query, embedder, collection, top_k=10, filter_kind=None,
                      cfg=None, where_extra=None):
        return list(dense)

    monkeypatch.setattr(agent, "_match_titles", lambda q, s, limit=4: [])
    monkeypatch.setattr(agent, "retrieve", fake_retrieve)

    res = agent.retrieve_rag("tt_fuse", "zebra", 3, None, {}, embedder, col)
    ids = [h["id"] for h in res["hits"]]
    assert ids[0] == "shared"


def test_bm25_leg_keeps_overlap_ids_ranked_without_refetching(kind_collection):
    _stub_bm25("tt_kind", {
        "f1": "dragon dragon sword",
        "n1": "dragon garden",
    })
    hits = agent._bm25_leg("tt_kind", "dragon", {"f1"}, kind_collection)
    by_id = {h["id"]: h for h in hits}
    assert by_id["f1"]["bm25_score"] is not None
    # Overlap hits ride on the dense side's document/metadata.
    assert by_id["f1"]["document"] == ""
    assert by_id["n1"]["document"] == "dragon garden"


# ─── 4. Stale BM25 postings are dropped ──────────────────────────────────────

def test_bm25_leg_skips_ids_missing_from_collection(kind_collection):
    _stub_bm25("tt_kind", {
        "ghost": "dragon dragon dragon",
        "n1": "dragon garden",
    })
    hits = agent._bm25_leg("tt_kind", "dragon", set(), kind_collection)
    assert [h["id"] for h in hits] == ["n1"]
    assert all(h["document"] for h in hits)


# ─── 5. Title mode runs BM25 + fusion + rerank ───────────────────────────────

def test_title_mode_runs_bm25_fusion_and_rerank(monkeypatch, embedder):
    col = _collection("tt_title", [
        ("d0", "dense zero", [1.0, 0.0, 0.0],
         {"kind": "nonfiction", "title": "Book A", "source": "s0"}),
    ])
    dense = [
        _hit("d0", "dense zero",
             {"kind": "nonfiction", "title": "Book A", "source": "s0"},
             distance=0.1),
        _hit("d1", "dense one",
             {"kind": "nonfiction", "title": "Book A", "source": "s1"},
             distance=0.2),
    ]
    retrieve_calls, bm25_calls, rerank_calls = [], [], []

    def fake_retrieve(query, embedder, collection, top_k=10, filter_kind=None,
                      cfg=None, where_extra=None):
        retrieve_calls.append(where_extra)
        return list(dense)

    def fake_bm25(set_name, query, skip_ids, collection, n=None,
                  backend="auto", cfg=None, filter_kind=None,
                  where_extra=None):
        bm25_calls.append(where_extra)
        return [
            _hit("d0", bm25_score=9.0),
            _hit("b1", "bm25 doc",
                 {"kind": "nonfiction", "title": "Book A", "source": "s2"},
                 bm25_score=5.0),
        ]

    def fake_rerank(hits, query, reranker, top_n=24):
        rerank_calls.append([h["id"] for h in hits])
        return hits, True

    def boom(*a, **k):
        raise AssertionError("multi-hop must not run in title mode")

    monkeypatch.setattr(agent, "_match_titles", lambda q, s, limit=4: ["Book A"])
    monkeypatch.setattr(agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent, "_bm25_leg", fake_bm25)
    monkeypatch.setattr(agent, "rerank_hits", fake_rerank)
    monkeypatch.setattr(agent, "retrieve_multi_hop", boom)
    monkeypatch.setattr(agent, "_get_bm25_index",
                        lambda *a, **k: {"bm25": None, "ids": [], "df": {}})

    res = agent.retrieve_rag("tt_title", "dragon", 3, None, {},
                             embedder, col, reranker=object())

    assert res["title_mode"] is True
    # Both legs were gated to the matched titles.
    assert retrieve_calls == [{"title": {"$in": ["Book A"]}}]
    assert bm25_calls == [{"title": {"$in": ["Book A"]}}]
    assert rerank_calls, "title mode must still rerank"
    assert {"d0", "d1", "b1"} <= {h["id"] for h in res["hits"]}
    assert res["hits"][0]["id"] == "d0"


def test_title_mode_falls_back_when_gated_dense_is_empty(monkeypatch, embedder):
    col = _collection("tt_title_fb", [
        ("d0", "dense zero", [1.0, 0.0, 0.0],
         {"kind": "nonfiction", "title": "Book A", "source": "s0"}),
    ])
    seen = []

    def fake_retrieve(query, embedder, collection, top_k=10, filter_kind=None,
                      cfg=None, where_extra=None):
        seen.append(where_extra)
        if where_extra is not None:
            return []
        return [_hit("d0", "dense zero",
                     {"kind": "nonfiction", "title": "Book A", "source": "s0"},
                     distance=0.1)]

    monkeypatch.setattr(agent, "_match_titles", lambda q, s, limit=4: ["Book A"])
    monkeypatch.setattr(agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent, "_bm25_leg",
                        lambda *a, **k: [])
    monkeypatch.setattr(agent, "_get_bm25_index",
                        lambda *a, **k: {"bm25": None, "ids": [], "df": {}})

    res = agent.retrieve_rag("tt_title_fb", "dragon", 3, None, {},
                             embedder, col)
    assert res["title_mode"] is False
    assert [h["id"] for h in res["hits"]] == ["d0"]
    assert seen == [{"title": {"$in": ["Book A"]}}, None]


# ─── 6. Top-k cap is configurable ────────────────────────────────────────────

def _record_pool(monkeypatch, embedder):
    calls = []

    def fake_retrieve(query, embedder, collection, top_k=10, filter_kind=None,
                      cfg=None, where_extra=None):
        calls.append(top_k)
        return []

    monkeypatch.setattr(agent, "_match_titles", lambda q, s, limit=4: [])
    monkeypatch.setattr(agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent, "_bm25_leg", lambda *a, **k: [])
    monkeypatch.setattr(agent, "_get_bm25_index",
                        lambda *a, **k: {"bm25": None, "ids": [], "df": {}})
    return calls


def test_top_k_cap_defaults_to_eight(monkeypatch, embedder):
    calls = _record_pool(monkeypatch, embedder)
    agent.retrieve_rag("tt_cap", "query", 20, None, {}, embedder, None)
    assert calls == [min(8 * 3, 30)]


def test_top_k_cap_reads_config_key(monkeypatch, embedder):
    calls = _record_pool(monkeypatch, embedder)
    agent.retrieve_rag("tt_cap", "query", 20, None,
                       {"retrieval_top_k_max": 3}, embedder, None)
    assert calls == [min(3 * 3, 30)]


# ─── 16. Empty collection: BM25 index cached as None ─────────────────────────

def test_empty_collection_caches_empty_bm25_index(monkeypatch):
    col = _collection("tt_empty", [])
    idx = agent._get_bm25_index("tt_empty", col, backend="bm25_chroma", cfg={})
    assert idx["bm25"] is None
    assert idx["ids"] == []
    assert agent._bm25_cache["tt_empty"]["bm25"] is None

    def boom(*a, **k):
        raise AssertionError("index rebuilt for an empty collection")

    monkeypatch.setattr(col, "get", boom)
    again = agent._get_bm25_index("tt_empty", col, backend="bm25_chroma", cfg={})
    assert again["bm25"] is None


def test_bm25_leg_returns_empty_for_none_index(kind_collection):
    agent._bm25_cache["tt_kind"] = {"bm25": None, "ids": [], "df": {}}
    assert agent._bm25_leg("tt_kind", "dragon", set(), kind_collection) == []
