"""Config secret masking and the POST /api/config sets validation."""

import json

SECRET = "sk-test-SECRET-1234567890"


def _read(path):
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_public_cfg_masks_key(server):
    out = server._public_cfg({"llm_api_key": SECRET})
    assert out["llm_api_key"] == server.API_KEY_MASK
    assert out["llm_api_key_set"] is True
    empty = server._public_cfg({})
    assert empty["llm_api_key"] is None
    assert empty["llm_api_key_set"] is False


def test_secret_never_returned_or_written_to_tracked_config(client, test_root):
    r = client.post("/api/config", json={"llm_api_key": SECRET})
    assert r.status_code == 200
    assert SECRET not in r.get_data(as_text=True)

    g = client.get("/api/config")
    assert g.status_code == 200
    assert SECRET not in g.get_data(as_text=True)
    assert g.get_json()["llm_api_key"] == "••••••••"

    # Secret lives only in the gitignored local overrides file.
    assert SECRET not in _read(test_root / "config.json")
    assert SECRET in _read(test_root / "config.local.json")


def test_posting_mask_back_does_not_overwrite_secret(client, test_root):
    client.post("/api/config", json={"llm_api_key": SECRET})
    r = client.post("/api/config", json={"llm_api_key": "••••••••",
                                         "llm_model": "other-model"})
    assert r.status_code == 200
    assert SECRET not in r.get_data(as_text=True)
    assert SECRET in _read(test_root / "config.local.json")
    assert json.loads(_read(test_root / "config.json"))["llm_model"] == "other-model"


def test_empty_key_clears_secret(client, test_root):
    client.post("/api/config", json={"llm_api_key": SECRET})
    client.post("/api/config", json={"llm_api_key": ""})
    assert SECRET not in _read(test_root / "config.local.json")


def test_non_object_body_rejected(client):
    r = client.post("/api/config", json=["not", "an", "object"])
    assert r.status_code == 400


def test_sets_must_be_mapping_of_objects(client):
    bad_bodies = [
        {"sets": "bad"},
        {"sets": ["x"]},
        {"sets": {"a": "notdict"}},
        {"sets": {"a": {"path": 5}}},
        {"sets": {"a": None}},
    ]
    for body in bad_bodies:
        r = client.post("/api/config", json=body)
        assert r.status_code == 400, body
        assert r.get_json()["ok"] is False


def test_valid_sets_stored_in_local_file_only(client, test_root):
    sets = {"books": {"path": "/srv/books", "kind": "local"}}
    r = client.post("/api/config", json={"sets": sets})
    assert r.status_code == 200
    assert json.loads(_read(test_root / "config.local.json"))["sets"] == sets
    assert "sets" not in json.loads(_read(test_root / "config.json"))
