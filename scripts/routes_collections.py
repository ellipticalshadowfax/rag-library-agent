"""Collection and set management routes.

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
    RAG_ROOT,
    _chroma_client,
    _drop_manifest_set,
    _rename_manifest_set,
    ingest_active,
    load_cfg,
    save_cfg,
)

bp = Blueprint("collections", __name__)


@bp.route("/api/collections")
def api_collections():
    client = __import__("chromadb", fromlist=["PersistentClient"]).PersistentClient(path=str(RAG_ROOT / "index"))
    cfg = load_cfg()
    out = []
    for col in client.list_collections():
        set_cfg = (cfg.get("sets") or {}).get(col.name, {})
        out.append({
            "name": col.name,
            "count": col.count(),
            "path": set_cfg.get("path", ""),
            "kind": set_cfg.get("kind", "local"),
        })
    return jsonify(out)


@bp.route("/api/sets")
def api_sets():
    cfg = load_cfg()
    client = _chroma_client()
    names = {c.name for c in client.list_collections()}
    out = []
    for name, info in (cfg.get("sets") or {}).items():
        out.append({
            "name": name,
            "path": info.get("path", ""),
            "kind": info.get("kind", "local"),
            "indexed": name in names,
        })
    return jsonify(out)


@bp.route("/api/sets", methods=["POST"])
def api_sets_save():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    path = (data.get("path") or "").strip()
    if not name or not path:
        return jsonify({"ok": False, "message": "Set name and directory are required."}), 400
    cfg = load_cfg()
    cfg.setdefault("sets", {})[name] = {"path": path, "kind": "local"}
    save_cfg(cfg)
    return jsonify({"ok": True, "message": f"Set '{name}' registered."})


@bp.route("/api/sets/<name>", methods=["DELETE"])
def api_sets_delete(name):
    if ingest_active():
        return jsonify({"ok": False, "message": "Ingest is running — wait for it to finish."}), 409
    cfg = load_cfg()
    if name in (cfg.get("sets") or {}):
        del cfg["sets"][name]
        save_cfg(cfg)
    return jsonify({"ok": True})


@bp.route("/api/collections/rename", methods=["POST"])
def api_collection_rename():
    if ingest_active():
        return jsonify({"ok": False, "message": "Ingest is running — cannot rename."}), 409
    data = request.get_json(force=True) or {}
    old = (data.get("from") or "").strip()
    new = (data.get("to") or "").strip()
    if not old or not new:
        return jsonify({"ok": False, "message": "Both current and new names are required."}), 400
    if old == new:
        return jsonify({"ok": False, "message": "New name is the same as the current name."}), 400
    client = _chroma_client()
    names = {c.name for c in client.list_collections()}
    if old not in names:
        return jsonify({"ok": False, "message": f"Collection '{old}' not found."}), 404
    if new in names:
        return jsonify({"ok": False, "message": f"Collection '{new}' already exists."}), 400
    try:
        client.get_collection(old).modify(name=new)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Rename failed: {e}"}), 500
    cfg = load_cfg()
    if old in (cfg.get("sets") or {}):
        cfg["sets"][new] = cfg["sets"].pop(old)
        save_cfg(cfg)
    try:
        _rename_manifest_set(old, new)
    except Exception as e:
        print(f"[collections/rename] manifest update failed: {e}")
    return jsonify({"ok": True, "message": f"Renamed '{old}' to '{new}'."})


@bp.route("/api/collections/<name>", methods=["DELETE"])
def api_collection_delete(name):
    if ingest_active():
        return jsonify({"ok": False, "message": "Ingest is running — wait for it to finish."}), 409
    client = _chroma_client()
    names = {c.name for c in client.list_collections()}
    if name not in names:
        return jsonify({"ok": False, "message": f"Collection '{name}' not found."}), 404
    try:
        client.delete_collection(name)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Delete failed: {e}"}), 500
    cfg = load_cfg()
    if name in (cfg.get("sets") or {}):
        del cfg["sets"][name]
        save_cfg(cfg)
    try:
        _drop_manifest_set(name)
    except Exception as e:
        print(f"[collections/delete] manifest update failed: {e}")
    return jsonify({"ok": True, "message": f"Deleted collection '{name}'."})
