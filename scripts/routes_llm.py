"""Embedding model and LLM configuration/self-host routes.

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
    EMBED_LOAD_LOCK,
    EMBED_LOAD_STATE,
    EMBED_MODELS,
    RAG_ROOT,
    SELFHOST_DL_STATE,
    SELFHOST_EST_MB,
    SELFHOST_GGUF,
    SELFHOST_PROC,
    SELFHOST_REPO,
    SELFHOST_URL,
    _CFG_LOCK,
    _LLM_PROBE_CACHE,
    _LLM_PROBE_TTL,
    _START_LOCK,
    _find_llama_server,
    _hf_cache_dir_mb,
    _llm_apply_locked,
    _llm_proc_alive,
    _load_embed_model,
    _public_cfg,
    check_llm,
    get_chat_client,
    load_cfg,
    save_cfg,
)
import server_core as _core

bp = Blueprint("llm", __name__)


@bp.route("/api/embed/models")
def api_embed_models():
    cfg = load_cfg()
    cur = cfg.get("embed_model")
    out = []
    for m in EMBED_MODELS:
        item = dict(m)
        item["current"] = (m["id"] == cur)
        item["cached"] = _hf_cache_dir_mb(m["id"]) > 0
        out.append(item)
    return jsonify(out)


@bp.route("/api/embed/load", methods=["POST"])
def api_embed_load():
    data = request.get_json(force=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"ok": False, "error": "No model specified."}), 400
    with EMBED_LOAD_LOCK:
        if EMBED_LOAD_STATE["running"]:
            return jsonify({"ok": False,
                            "error": "A model load is already in progress."}), 409
    device = data.get("device") or "auto"
    threading.Thread(target=_load_embed_model, args=(model, device),
                     daemon=True).start()
    return jsonify({"ok": True})


@bp.route("/api/embed/load/status")
def api_embed_load_status():
    with EMBED_LOAD_LOCK:
        return jsonify(dict(EMBED_LOAD_STATE))


@bp.route("/api/embed/apply", methods=["POST"])
def api_embed_apply():
    data = request.get_json(force=True) or {}
    cfg = load_cfg()
    model = (data.get("model") or "").strip()
    device = data.get("device") or cfg.get("embed_device", "auto")
    dim = data.get("dim")
    if model:
        cfg["embed_model"] = model
    if device:
        cfg["embed_device"] = device
    if dim:
        try:
            cfg["embed_dim"] = int(dim)
        except (TypeError, ValueError):
            pass
    save_cfg(cfg)
    # Chat embedder is cached; drop it so the new model takes effect.
    _core._chat_embedder = None
    return jsonify({"ok": True, "config": _public_cfg(cfg), "reindex": True})


@bp.route("/api/llm/status")
def api_llm_status():
    return jsonify(check_llm())


@bp.route("/api/llm/test", methods=["POST"])
def api_llm_test():
    cfg = load_cfg()
    model = cfg.get("llm_model", "default")
    now = time.time()
    c = _LLM_PROBE_CACHE
    if c["result"] is not None and c["model"] == model \
            and (now - c["at"]) < _LLM_PROBE_TTL:
        return jsonify(c["result"])
    result = {"ok": False, "loaded": False, "model": model, "error": None}
    try:
        client = get_chat_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        text = (resp.choices[0].message.content or "").strip()
        result.update(ok=True, loaded=True, sample=text[:80])
    except Exception as e:
        result["error"] = str(e)
    _LLM_PROBE_CACHE.update(at=now, model=model, result=result)
    return jsonify(result)


@bp.route("/api/llm/apply", methods=["POST"])
def api_llm_apply():
    data = request.get_json(force=True) or {}
    with _CFG_LOCK:
        return _llm_apply_locked(data)


@bp.route("/api/llm/selfhost")
def api_llm_selfhost():
    binary = _find_llama_server()
    models_dir = RAG_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    gguflist = [f.name for f in models_dir.glob("*.gguf")]
    return jsonify({
        "available": binary is not None,
        "binary": binary,
        "dir": str(models_dir),
        "models": gguflist,
        "candidate": {
            "file": SELFHOST_GGUF, "repo": SELFHOST_REPO,
            "url": SELFHOST_URL, "est_mb": SELFHOST_EST_MB,
        },
    })


@bp.route("/api/llm/download", methods=["POST"])
def api_llm_download():
    if not _find_llama_server():
        return jsonify({"ok": False,
                        "error": "llama-server not found on this machine. "
                                 "Self-hosted LLM is not available here."}), 400
    if SELFHOST_DL_STATE["running"]:
        return jsonify({"ok": False,
                        "error": "A download is already in progress."}), 409
    models_dir = RAG_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    def _dl():
        dest = models_dir / SELFHOST_GGUF
        import urllib.request
        SELFHOST_DL_STATE.update(running=True, phase="downloading",
                                 progress=0, downloaded_mb=0, error=None)
        try:
            req = urllib.request.Request(SELFHOST_URL, method="GET")
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    SELFHOST_DL_STATE.update(
                        downloaded_mb=got // (1024 * 1024),
                        progress=int(100 * got / total) if total else -1)
            SELFHOST_DL_STATE.update(running=False, phase="done", progress=100,
                                     downloaded_mb=got // (1024 * 1024))
        except Exception as e:
            SELFHOST_DL_STATE.update(running=False, phase="error", error=str(e))

    threading.Thread(target=_dl, daemon=True).start()
    return jsonify({"ok": True})


@bp.route("/api/llm/download/status")
def api_llm_download_status():
    return jsonify(SELFHOST_DL_STATE)


@bp.route("/api/llm/start", methods=["POST"])
def api_llm_start():
    binary = _find_llama_server()
    if not binary:
        return jsonify({"ok": False,
                        "error": "llama-server not found on this machine."}), 400
    models_dir = RAG_ROOT / "models"
    gguflist = sorted(models_dir.glob("*.gguf"))
    if not gguflist:
        return jsonify({"ok": False,
                        "error": "No GGUF model in models/ yet."}), 400
    gguf = gguflist[0]
    # Check-and-spawn must be atomic, or two quick requests both see "not running".
    with _START_LOCK:
        if _llm_proc_alive():
            return jsonify({"ok": False, "error": "llama-server already running."}), 409
        log_path = RAG_ROOT / "llama-server.log"
        with open(log_path, "w") as logf:
            proc = subprocess.Popen([binary, "-m", str(gguf), "-c", "8192"],
                                    stdout=logf, stderr=subprocess.STDOUT)
        _core._LLM_PROC = proc
        SELFHOST_PROC["pid"] = proc.pid
    return jsonify({"ok": True, "pid": proc.pid, "model": str(gguf)})


@bp.route("/api/llm/control", methods=["POST"])
def api_llm_control():
    data = request.get_json(force=True) or {}
    action = data.get("action")
    if not _llm_proc_alive():
        return jsonify({"ok": False, "error": "llama-server is not running."}), 404
    proc = _core._LLM_PROC
    if proc is None:  # liveness check above guaranteed otherwise
        return jsonify({"ok": False, "error": "llama-server is not running."}), 404
    pid = SELFHOST_PROC["pid"]
    if action == "stop":
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        _core._LLM_PROC = None
        SELFHOST_PROC["pid"] = None
        return jsonify({"ok": True, "message": "llama-server stopped."})
    return jsonify({"ok": False, "error": "Unknown action."}), 400
