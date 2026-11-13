"""Config, system, health, filesystem and first-run setup routes.

Registered on the app by server.py. The origin guard stays on the app.
"""

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request, send_from_directory, Response
from _paths import (LOCAL_KEYS, SECRET_KEYS, atomic_write_json, cuda_available,
                    rag_root, resolve_device)
import agent
from agent import CollectionNotFound
import chat_store
import ingest as ING
import scan as SCAN
import catalog as CAT
import study as STUDY
import quiz_build
import quiz_export
import quiz_grade
import quiz_plan
import quiz_syllabus
import quiz_store
from flask import Blueprint

from server_core import (
    API_KEY_MASK,
    CONFIG_META,
    EMBED_MODELS,
    _CFG_LOCK,
    _config_writable_keys,
    _public_cfg,
    check_llm,
    detect_source_dirs,
    load_cfg,
    save_cfg,
)
import server_core as _core

bp = Blueprint("config", __name__)


@bp.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify(_public_cfg(load_cfg()))


@bp.route("/api/config/schema", methods=["GET"])
def api_config_schema():
    """Return the full {key: meta} registry for tools/agents to introspect."""
    return jsonify(CONFIG_META)


@bp.route("/api/config", methods=["POST"])
def api_config_set():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "JSON object expected."}), 400
    if "sets" in data:
        sets = data["sets"]
        if not isinstance(sets, dict) or not all(
                isinstance(k, str) and isinstance(v, dict)
                and isinstance(v.get("path", ""), str)
                for k, v in sets.items()):
            return jsonify({"ok": False,
                            "error": "'sets' must map names to {path, kind} objects."}), 400
    with _CFG_LOCK:
        cfg = load_cfg()
        # Only allow whitelisted keys to update; the allowlist is derived from
        # the CONFIG_META registry so new keys can never drift out of sync.
        allowed = _config_writable_keys()
        for k in allowed:
            if k not in data:
                continue
            if k in SECRET_KEYS and str(data[k]).strip() == API_KEY_MASK:
                # GET /api/config returns a masked key; a read-modify-write
                # client must not overwrite the real secret with the mask.
                continue
            cfg[k] = data[k]
        if "sets" in data:
            cfg["sets"] = data["sets"]
        save_cfg(cfg)
    return jsonify({"ok": True, "config": _public_cfg(cfg)})


@bp.route("/api/system")
def api_system():
    llm = check_llm()
    cfg = load_cfg()
    gpu = cuda_available()
    return jsonify({
        "llm": llm,
        "lmstudio": llm,  # backward-compatible alias
        "source_dirs": detect_source_dirs(),
        "cuda": gpu,                      # usable CUDA present (hw + torch build)
        "gpu_available": gpu,
        "embed_device_resolved": resolve_device(cfg.get("embed_device")),
        "drives": detect_source_dirs(),
    })


# ─── First-run wizard ────────────────────────────────────────────────────────

@bp.route("/api/setup/status")
def api_setup_status():
    cfg = load_cfg()
    return jsonify({
        "needs_setup": not cfg.get("setup_complete", False),
        "llm": check_llm(),
        "embed": {
            "model": cfg.get("embed_model"),
            "device": cfg.get("embed_device"),
            "dim": cfg.get("embed_dim"),
            "models": [dict(m) for m in EMBED_MODELS],
        },
        "sets": cfg.get("sets", {}),
    })


@bp.route("/api/setup/finish", methods=["POST"])
def api_setup_finish():
    data = request.get_json(force=True) or {}
    cfg = load_cfg()
    if "llm_base_url" in data:
        cfg["llm_base_url"] = str(data["llm_base_url"]).strip()
    if "llm_model" in data and data["llm_model"]:
        cfg["llm_model"] = str(data["llm_model"]).strip()
    if "llm_api_key" in data:
        key = str(data["llm_api_key"]).strip()
        if key and key != API_KEY_MASK:
            cfg["llm_api_key"] = key
        elif not key:
            cfg["llm_api_key"] = None
    if "embed_model" in data:
        cfg["embed_model"] = str(data["embed_model"]).strip()
    if "embed_device" in data:
        cfg["embed_device"] = str(data["embed_device"]).strip()
    if "embed_dim" in data and data["embed_dim"]:
        try:
            cfg["embed_dim"] = int(data["embed_dim"])
        except (TypeError, ValueError):
            pass
    if "sets" in data and isinstance(data["sets"], dict):
        cfg["sets"] = data["sets"]
    cfg["setup_complete"] = True
    save_cfg(cfg)
    _core._chat_client = None
    _core._chat_embedder = None
    return jsonify({"ok": True, "config": _public_cfg(cfg)})


@bp.route("/api/health")
def api_health():
    return jsonify({"ok": True})


@bp.route("/api/fs")
def api_fs():
    """List subdirectories of a given path (for the folder picker dialog)."""
    d = request.args.get("dir", "/")
    d = os.path.abspath(os.path.expanduser(d))
    if not os.path.isdir(d):
        return jsonify({"error": f"Not a directory: {d}"}), 400
    try:
        entries = sorted(os.listdir(d))
    except PermissionError:
        return jsonify({"error": f"Permission denied: {d}"}), 403
    except OSError as e:
        return jsonify({"error": f"Unable to read: {d} ({e})"}), 500

    dirs = []
    for e in entries:
        if e.startswith("."):
            continue
        p = os.path.join(d, e)
        if os.path.isdir(p):
            dirs.append({"name": e, "path": p})

    parent = os.path.dirname(d.rstrip("/")) or "/"
    return jsonify({"dir": d, "parent": parent, "dirs": dirs})
