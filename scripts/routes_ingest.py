"""Scan, ingest and OCR job routes.

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
    INGEST_STATE,
    OCR_STATE,
    RAG_ROOT,
    SCAN_STATE,
    _stop_watchdog,
    live_ingest_pid,
    live_ocr_pid,
    read_ingest_status,
    read_ocr_status,
    run_scan,
    start_ingest,
    start_ocr,
)

bp = Blueprint("ingest", __name__)


@bp.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json(force=True) or {}
    target = data.get("target", "")
    if not target:
        return jsonify({"error": "No target directory"}), 400
    if SCAN_STATE["running"]:
        return jsonify({"error": "Scan already running"}), 409
    threading.Thread(target=run_scan, args=(target,), daemon=True).start()
    return jsonify({"ok": True})


@bp.route("/api/scan/status")
def api_scan_status():
    return jsonify(SCAN_STATE)


@bp.route("/api/ingest", methods=["POST"])
def api_ingest():
    data = request.get_json(force=True) or {}
    ok, msg = start_ingest(
        target=data.get("target", ""),
        set_name=data.get("set", ""),
        force=data.get("force", False),
        only=data.get("only"),
    )
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg}), code


@bp.route("/api/ingest/status")
def api_ingest_status():
    return jsonify(read_ingest_status())


@bp.route("/api/ingest/control", methods=["POST"])
def api_ingest_control():
    data = request.get_json(force=True) or {}
    action = data.get("action")
    pid = live_ingest_pid()
    if not pid:
        return jsonify({"ok": False, "message": "No ingest is running."}), 404
    try:
        if action == "pause":
            os.kill(pid, signal.SIGSTOP)
            INGEST_STATE["paused"] = True
            return jsonify({"ok": True, "paused": True,
                            "message": "Ingest paused (process suspended)."})
        if action == "resume":
            os.kill(pid, signal.SIGCONT)
            INGEST_STATE["paused"] = False
            return jsonify({"ok": True, "paused": False,
                            "message": "Ingest resumed."})
        if action == "stop":
            os.kill(pid, signal.SIGINT)
            INGEST_STATE.update(paused=False, stopping=True,
                                message="Stopping \u2014 finishing current file then exiting\u2026")
            threading.Thread(
                target=_stop_watchdog,
                args=(pid, INGEST_STATE, RAG_ROOT / ".ingest.lock"),
                daemon=True,
            ).start()
            return jsonify({"ok": True,
                            "message": "Stop requested \u2014 finishing current step then exiting."})
    except ProcessLookupError:
        return jsonify({"ok": False, "message": "Process already exited."}), 404
    except PermissionError:
        return jsonify({"ok": False, "message": "No permission to control the process."}), 403
    return jsonify({"ok": False, "message": "Unknown action."}), 400


@bp.route("/api/ocr", methods=["POST"])
def api_ocr():
    data = request.get_json(force=True) or {}
    ok, msg = start_ocr(
        target=data.get("target", ""),
        mode=data.get("mode", "merge"),
        force=data.get("force", False),
        only=data.get("only"),
        languages=data.get("languages"),
        backend=data.get("backend"),
    )
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg}), code


@bp.route("/api/ocr/status")
def api_ocr_status():
    return jsonify(read_ocr_status())


@bp.route("/api/ocr/control", methods=["POST"])
def api_ocr_control():
    data = request.get_json(force=True) or {}
    action = data.get("action")
    pid = live_ocr_pid()
    if not pid:
        return jsonify({"ok": False, "message": "No OCR job is running."}), 404
    try:
        if action == "pause":
            os.kill(pid, signal.SIGSTOP)
            OCR_STATE["paused"] = True
            return jsonify({"ok": True, "paused": True, "message": "OCR paused."})
        if action == "resume":
            os.kill(pid, signal.SIGCONT)
            OCR_STATE["paused"] = False
            return jsonify({"ok": True, "paused": False, "message": "OCR resumed."})
        if action == "stop":
            os.kill(pid, signal.SIGINT)
            OCR_STATE["paused"] = False
            threading.Thread(
                target=_stop_watchdog,
                args=(pid, OCR_STATE, RAG_ROOT / ".ocr.lock"),
                daemon=True,
            ).start()
            return jsonify({"ok": True,
                            "message": "Stop requested \u2014 finishing current step then exiting."})
    except ProcessLookupError:
        return jsonify({"ok": False, "message": "Process already exited."}), 404
    except PermissionError:
        return jsonify({"ok": False, "message": "No permission to control the process."}), 403
    return jsonify({"ok": False, "message": "Unknown action."}), 400
