#!/usr/bin/env python3
"""RAG Web App - Setup, Scan, Ingest, and Chat in one browser UI.

Run:  source .venv/bin/activate && python scripts/server.py  (defaults to port 5000)
Open: http://localhost:5000
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Skip HuggingFace remote checks; models cached from first-run

from flask import Flask, jsonify, request, send_from_directory, Response

from _paths import (LOCAL_KEYS, SECRET_KEYS, atomic_write_json, cuda_available,
                    rag_root, resolve_device)
import agent
from agent import CollectionNotFound

RAG_ROOT = rag_root()
SCRIPTS_DIR = RAG_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

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

from server_core import *  # noqa: F401,F403  (re-exported for callers)
from server_core import (  # noqa: F401  (underscore names used by tests)
    _as_int, _public_cfg,
)


app = Flask(__name__, static_folder=str(RAG_ROOT / "web"), static_url_path="/web")

# ─── Request origin guard ────────────────────────────────────────────────────
# The API has no auth and is meant for local use. Without this guard any
# website open in the user's browser could call it (CORS-less simple POSTs and
# DNS rebinding both reach loopback). We therefore:
#   * require the Host header to name a loopback host (or one listed in
#     RAG_ALLOWED_HOSTS, comma-separated) — defeats DNS rebinding;
#   * reject requests carrying an Origin whose host isn't allowed — defeats
#     cross-site fetch/XHR from other pages.
# Non-browser clients (curl, MCP, OpenAI-compatible tools) send no Origin and
# are unaffected as long as they use an allowed Host.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _allowed_hosts() -> set:
    extra = {h.strip().lower() for h in
             os.environ.get("RAG_ALLOWED_HOSTS", "").split(",") if h.strip()}
    bind = os.environ.get("RAG_HOST", "127.0.0.1").strip().lower()
    return _LOOPBACK_HOSTS | extra | ({bind} if bind not in ("0.0.0.0", "::") else set())


def _hostname(netloc: str) -> str:
    netloc = (netloc or "").strip().lower()
    if netloc.startswith("["):  # [::1]:5000
        return netloc.split("]")[0] + "]"
    return netloc.split(":")[0]


@app.before_request
def _guard_origin():
    allowed = _allowed_hosts()
    if _hostname(request.host) not in allowed:
        return jsonify({"error": "Host not allowed."}), 403
    origin = request.headers.get("Origin")
    if origin and origin != "null":
        from urllib.parse import urlsplit
        if _hostname(urlsplit(origin).netloc) not in allowed:
            return jsonify({"error": "Cross-origin requests are not allowed."}), 403
    elif origin == "null":
        return jsonify({"error": "Cross-origin requests are not allowed."}), 403

app.config["JSON_AS_ASCII"] = False
# Cap request-body size so a huge/streamed POST can't exhaust memory.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MiB

# ─── Blueprints (route modules) ─────────────────────────────────────────────
from routes_config import bp as config_bp
from routes_llm import bp as llm_bp
from routes_ingest import bp as ingest_bp
from routes_quiz import bp as quiz_bp
from routes_chat import bp as chat_bp
from routes_collections import bp as collections_bp

app.register_blueprint(config_bp)
app.register_blueprint(llm_bp)
app.register_blueprint(ingest_bp)
app.register_blueprint(quiz_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(collections_bp)



# ─── Flask routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(RAG_ROOT / "web", "index.html")


@app.route("/web/<path:filename>")
def static_files(filename):
    return send_from_directory(RAG_ROOT / "web", filename)


if __name__ == "__main__":
    port = int(os.environ.get("RAG_PORT", 5000))
    host = os.environ.get("RAG_HOST", "127.0.0.1")
    cfg = load_cfg()
    print(f"\n  RAG Web App")
    print(f"  -----------")
    print(f"  Open:  http://{host}:{port}")
    print(f"  Embedder: {cfg.get('embed_model')}")
    print(f"  LLM API: {cfg.get('llm_base_url')}  -> {check_llm()}")
    print(f"  Config: {RAG_ROOT / 'config.json'}")
    print(f"  Index:  {RAG_ROOT / 'index'}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
