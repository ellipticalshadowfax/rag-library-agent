"""Remote embedder / reranker backends (_paths.py). No real network: every
test monkeypatches ``requests.post``."""

import pytest
import requests

import _paths
from _paths import (OllamaRemoteEmbedder, RemoteReranker, build_embedder,
                    build_reranker, embedder_identity, resolve_embed_prefixes)


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._p = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._p


def _embed_post(calls, dim=3):
    def post(url, json=None, timeout=None, **kw):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResp({"embeddings": [[0.5] * dim for _ in json["input"]]})
    return post


# ── 7: prefixes ─────────────────────────────────────────────────────────────
def test_prefix_table_and_overrides():
    assert resolve_embed_prefixes({}, "intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    q, p = resolve_embed_prefixes({}, "Qwen3-Embedding:0.6b")
    assert q.startswith("Instruct: ") and q.endswith("\nQuery: ") and p == ""
    assert resolve_embed_prefixes({}, "bge-m3") == ("", "")
    cfg = {"embed_query_prefix": None, "embed_passage_prefix": ""}
    assert resolve_embed_prefixes(cfg, "e5-x") == ("query: ", "")  # '' is explicit
    cfg = {"embed_query_prefix": "Q:", "embed_passage_prefix": "P:"}
    assert resolve_embed_prefixes(cfg, "bge-m3") == ("Q:", "P:")


def test_encode_applies_prompt_name_prefix(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", _embed_post(calls))
    e = OllamaRemoteEmbedder("http://h:1", "m", query_prefix="query: ", passage_prefix="passage: ")
    e.encode(["a", "b"], prompt_name="query")
    e.encode("c", prompt_name="passage")
    e.encode("d", prompt_name=None)
    e.encode("e", prompt_name="other")
    assert calls[0]["json"]["input"] == ["query: a", "query: b"]
    assert calls[1]["json"]["input"] == ["passage: c"]
    assert calls[2]["json"]["input"] == ["d"]
    assert calls[3]["json"]["input"] == ["e"]
    assert e.prompts == {"query": "query: ", "passage": "passage: "}


def test_build_embedder_wires_prefixes_timeout_and_no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network in __init__")
    monkeypatch.setattr(requests, "post", boom)
    e = build_embedder({"embed_backend": "ollama", "embed_model": "multilingual-e5-small",
                        "embed_api_url": "http://x:1", "embed_dim": 384, "embed_timeout": 7})
    assert e.prompts == {"query": "query: ", "passage": "passage: "}
    assert e.timeout == 7
    assert e.query_prefix == "query: "


def test_identity_remote_and_embedder_identity():
    cfg = {"embed_backend": "ollama", "embed_model": "qwen3-embedding:0.6b",
           "embed_dim": 1024, "embed_passage_prefix": "P "}
    ident = embedder_identity(cfg)
    assert set(ident) == {"backend", "model", "dim", "query_prefix", "passage_prefix"}
    assert ident["backend"] == "ollama" and ident["dim"] == 1024
    assert ident["passage_prefix"] == "P " and ident["query_prefix"].startswith("Instruct:")
    local = embedder_identity({"embed_model": "intfloat/multilingual-e5-small", "embed_dim": 384})
    assert local == {"backend": "local", "model": "intfloat/multilingual-e5-small",
                     "dim": 384, "query_prefix": "query: ", "passage_prefix": "passage: "}

    class M:
        prompts = {"query": "qq", "passage": "pp"}
    got = embedder_identity({"embed_model": "zzz", "embed_dim": 8}, model=M())
    assert got["query_prefix"] == "qq" and got["passage_prefix"] == "pp"


def test_embedder_identity_probes_real_dim(monkeypatch):
    monkeypatch.setattr(requests, "post", _embed_post([], dim=5))
    e = OllamaRemoteEmbedder("http://h:1", "m", dim=1024)
    assert e.identity["dim"] == 5


# ── 9: lazy dim probe ───────────────────────────────────────────────────────
def test_dim_probe_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", _embed_post(calls, dim=7))
    e = OllamaRemoteEmbedder("http://h:1", "m", dim=1024)
    assert calls == []
    assert e.get_sentence_embedding_dimension() == 7
    assert e.get_embedding_dimension() == 7
    assert len(calls) == 1


def test_dim_probe_failure_falls_back(monkeypatch):
    def post(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "post", post)
    e = OllamaRemoteEmbedder("http://h:1", "m", dim=384, sleep=lambda s: None)
    assert e.get_sentence_embedding_dimension() == 384


# ── 12: retry / timeouts / breaker ──────────────────────────────────────────
def test_embedder_retries_connection_errors_with_backoff(monkeypatch):
    sleeps, n = [], {"c": 0}

    def post(url, json=None, timeout=None, **kw):
        n["c"] += 1
        if n["c"] < 3:
            raise requests.ConnectionError("x")
        return FakeResp({"embeddings": [[1.0, 2.0]]})
    monkeypatch.setattr(requests, "post", post)
    e = OllamaRemoteEmbedder("http://h:1", "m", sleep=sleeps.append)
    assert e.encode("a").tolist() == [1.0, 2.0]
    assert n["c"] == 3 and sleeps == [0.5, 1.0]


def test_embedder_retries_5xx_then_raises(monkeypatch):
    sleeps, n = [], {"c": 0}

    def post(*a, **k):
        n["c"] += 1
        return FakeResp(status=503)
    monkeypatch.setattr(requests, "post", post)
    e = OllamaRemoteEmbedder("http://h:1", "m", sleep=sleeps.append)
    with pytest.raises(requests.HTTPError):
        e.encode("a")
    assert n["c"] == 3 and sleeps == [0.5, 1.0]


def test_embedder_never_retries_4xx(monkeypatch):
    sleeps, n = [], {"c": 0}

    def post(*a, **k):
        n["c"] += 1
        return FakeResp(status=400)
    monkeypatch.setattr(requests, "post", post)
    e = OllamaRemoteEmbedder("http://h:1", "m", sleep=sleeps.append)
    with pytest.raises(requests.HTTPError):
        e.encode("a")
    assert n["c"] == 1 and sleeps == []


def test_reranker_circuit_breaker(monkeypatch):
    now = {"t": 100.0}
    n = {"c": 0, "fail": True}

    def post(url, json=None, timeout=None, **kw):
        n["c"] += 1
        n["timeout"] = timeout
        if n["fail"]:
            raise requests.ConnectionError("down")
        return FakeResp({"results": [{"index": 0, "relevance_score": 1.0}]})
    monkeypatch.setattr(requests, "post", post)
    r = RemoteReranker("http://h:2", timeout=15, cooldown=30, clock=lambda: now["t"])
    with pytest.raises(requests.ConnectionError):
        r.predict([("q", "d")])
    assert n["timeout"] == 15
    with pytest.raises(Exception):
        r.predict([("q", "d")])
    assert n["c"] == 1  # short-circuited, no network
    now["t"] += 31
    n["fail"] = False
    assert r.predict([("q", "d")]) == [1.0]
    assert n["c"] == 2
    n["fail"] = True
    now["t"] += 1
    with pytest.raises(requests.ConnectionError):
        r.predict([("q", "d")])  # breaker was reset by success -> hits network


# ── 13: missing index fill ──────────────────────────────────────────────────
def test_reranker_missing_fill_is_below_min(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(
        {"results": [{"index": 0, "relevance_score": -8.0},
                     {"index": 2, "relevance_score": -3.0}]}))
    r = RemoteReranker("http://h:2")
    assert r.predict([("q", "a"), ("q", "b"), ("q", "c")]) == [-8.0, -9.0, -3.0]


def test_reranker_empty_results_raises_valueerror(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp({"results": []}))
    r = RemoteReranker("http://h:2")
    with pytest.raises(ValueError):
        r.predict([("q", "a")])


def test_build_reranker_remote_config(monkeypatch):
    r = build_reranker({"rerank_backend": "remote", "rerank_api_url": "http://z:9",
                        "rerank_timeout": 4, "rerank_cooldown": 9})
    assert r.timeout == 4 and r.cooldown == 9 and r.base_url == "http://z:9"
    r = build_reranker({"rerank_backend": "remote"})
    assert r.timeout == 15 and r.cooldown == 30
