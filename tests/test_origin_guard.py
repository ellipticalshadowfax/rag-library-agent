"""Origin/Host guard in scripts/server.py (DNS rebinding and cross-site defence)."""

import pytest


def test_loopback_host_and_no_origin_allowed(client):
    assert client.get("/api/config").status_code == 200


@pytest.mark.parametrize("base", [
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://[::1]:5000",
])
def test_loopback_hosts_allowed(client, base):
    assert client.get("/api/config", base_url=base).status_code == 200


def test_foreign_host_rejected(client):
    r = client.get("/api/config", base_url="http://evil.example:5000")
    assert r.status_code == 403


def test_foreign_host_rejected_even_with_loopback_origin(client):
    r = client.get("/api/config", base_url="http://evil.example:5000",
                   headers={"Origin": "http://localhost:5000"})
    assert r.status_code == 403


def test_foreign_origin_rejected(client):
    r = client.post("/api/config", json={"llm_model": "x"},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_origin_null_rejected(client):
    r = client.get("/api/config", headers={"Origin": "null"})
    assert r.status_code == 403


def test_loopback_origin_allowed(client):
    r = client.get("/api/config", headers={"Origin": "http://localhost:5000"})
    assert r.status_code == 200


def test_rag_allowed_hosts_extension_honoured(client, monkeypatch):
    monkeypatch.setenv("RAG_ALLOWED_HOSTS", "lab.internal, other.internal")
    assert client.get("/api/config",
                      base_url="http://lab.internal:5000").status_code == 200
    r = client.get("/api/config", base_url="http://lab.internal:5000",
                   headers={"Origin": "http://other.internal:5000"})
    assert r.status_code == 200


def test_rag_allowed_hosts_does_not_open_unlisted_hosts(client, monkeypatch):
    monkeypatch.setenv("RAG_ALLOWED_HOSTS", "lab.internal")
    r = client.get("/api/config", base_url="http://evil.example:5000")
    assert r.status_code == 403


def test_hostname_parsing(server):
    assert server._hostname("[::1]:5000") == "[::1]"
    assert server._hostname("LOCALHOST:5000") == "localhost"
    assert server._hostname("127.0.0.1") == "127.0.0.1"
    assert server._hostname("") == ""
