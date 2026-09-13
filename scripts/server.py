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

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for embeddings

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

from _paths import SECRET_KEYS, rag_root

RAG_ROOT = rag_root()
SCRIPTS_DIR = RAG_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import chat_store
import ingest as ING
import scan as SCAN

app = Flask(__name__, static_folder=str(RAG_ROOT / "web"), static_url_path="/web")
CORS(app)

app.config["JSON_AS_ASCII"] = False

# ─── Shared ingest progress state ────────────────────────────────────────────
INGEST_STATE = {
    "running": False,
    "paused": False,
    "pid": None,
    "set_name": None,
    "target": None,
    "total": 0,
    "current": 0,
    "done": 0,
    "skipped": 0,
    "errors": 0,
    "message": "",
    "lines": [],          # recent progress lines
    "finished": False,
    "completed": False,
    "started_at": None,
}

# Live Popen for ingest launched from this server (same rationale as _OCR_PROC:
# poll() reaps the child so a zombie can't keep "running" welded to true).
_INGEST_PROC = None


# ─── Shared OCR progress state ───────────────────────────────────────────────
OCR_STATE = {
    "running": False,
    "paused": False,
    "pid": None,
    "target": None,
    "mode": None,
    "total": 0,
    "current": 0,
    "done": 0,
    "skipped": 0,
    "errors": 0,
    "lines": [],
    "finished": False,
    "completed": False,
    "started_at": None,
}

# The live Popen for OCR launched from this server. Kept OUT of OCR_STATE so it
# can't be sent to the client; poll() is what reaps a finished child (a zombie
# would otherwise keep "running" true forever because os.kill(zombie, 0) works).
_OCR_PROC = None


# ─── Config helpers ──────────────────────────────────────────────────────────

def load_cfg():
    return ING.load_config()


def save_cfg(cfg):
    # Secrets (e.g. llm_api_key) must never be written to the git-tracked
    # config.json; they go to the gitignored config.local.json instead.
    public = {k: v for k, v in cfg.items() if k not in SECRET_KEYS}
    secrets = {k: v for k, v in cfg.items() if k in SECRET_KEYS}
    _write_json(RAG_ROOT / "config.json", public)
    if secrets:
        _save_local_overrides(secrets)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_local_overrides(updates: dict):
    """Merge ``updates`` into config.local.json, dropping keys set to None."""
    local_path = RAG_ROOT / "config.local.json"
    overrides = {}
    if local_path.exists():
        try:
            with open(local_path) as f:
                overrides = json.load(f) or {}
        except Exception:
            overrides = {}
    for k, v in updates.items():
        if v is None or v == "":
            overrides.pop(k, None)
        else:
            overrides[k] = v
    _write_json(local_path, overrides)


# Client-facing placeholder for a set API key. The real key is never returned.
API_KEY_MASK = "••••••••"


def _public_cfg(cfg: dict) -> dict:
    """Return a copy of cfg safe to send to the client (secret masked)."""
    out = dict(cfg)
    has_key = bool(out.get("llm_api_key"))
    out["llm_api_key"] = API_KEY_MASK if has_key else None
    out["llm_api_key_set"] = has_key
    return out


def default_cfg():
    # NOTE: This is only the fallback when config.json is missing; the live
    # settings live in config.json and are what the Setup tab edits. The RAG
    # *tuning* keys (grouped below) are the ones that most affect retrieval
    # quality vs. latency/context size:
    #   chunk_tokens / chunk_overlap  — indexing granularity; smaller = finer
    #                                   retrieval, more chunks to embed.
    #   parent_tokens                 — section size for parent_child generation.
    #   retrieval_top_k               — chunks surfaced in the prompt (capped 8).
    #   relevance_threshold           — low-relevance guard score floor.
    #   max_retrieval_hops            — extra LLM-driven search rounds.
    #   llm_temperature / llm_max_tokens — generation creativity & length caps.
    #   rerank_enabled / rerank_model — cross-encoder re-scoring (adds latency).
    return {
        "embed_model": "intfloat/multilingual-e5-small",
        "embed_device": "cpu",
        "embed_dim": 384,
        "chunk_tokens": 330,
        "chunk_overlap": 60,
        "chunking_strategy": "flat",
        "parent_tokens": 1200,
        "agentic_enabled": True,
        "agentic_max_steps": 3,
        "agentic_strategy": "auto",
        "llm_base_url": "http://localhost:1234/v1",
        "llm_model": "default",
        "llm_api_key": None,
        "llm_temperature": 0.3,
        "llm_max_tokens": 2048,
        "setup_complete": False,
        "retrieval_top_k": 10,
        "relevance_threshold": 0.80,
        "max_retrieval_hops": 2,
        "retrieval_hops_driver": "llm",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_enabled": True,
        "fiction_tags": ["Fiction", "Short Stories", "Literary"],
        "ocr_enabled": False,
        "ocr_merge": True,
        "ocr_backend": "rapidocr",
        "ocr_char_threshold": 50,
        "ocr_languages": ["en"],
        "sets": {},
    }


# ─── System / setup checks ───────────────────────────────────────────────────

def detect_source_dirs():
    """Return likely library directories (Calibre libraries w/ metadata.db)."""
    candidates = []
    for base in ["/media", "/home", os.path.expanduser("~")]:
        try:
            for p in Path(base).iterdir():
                if p.is_dir():
                    candidates.append(str(p))
        except Exception:
            pass
    return sorted(set(candidates))


def check_llm():
    """Check if the configured LLM API (LM Studio / llama.cpp / cloud) is
    reachable and list its models."""
    cfg = load_cfg()
    import urllib.request
    url = cfg["llm_base_url"].rstrip("/") + "/models"
    headers = {}
    api_key = cfg.get("llm_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode())
        models = [m.get("id", "") for m in data.get("data", [])]
        return {"online": True, "models": models, "url": cfg["llm_base_url"]}
    except Exception as e:
        return {"online": False, "models": [], "error": str(e), "url": cfg["llm_base_url"]}


# Backward-compatible alias used in a few places.
check_lmstudio = check_llm


# ─── Scan (non-blocking in thread) ───────────────────────────────────────────

SCAN_STATE = {"running": False, "result": None, "error": None}


def run_scan(target):
    SCAN_STATE["running"] = True
    SCAN_STATE["result"] = None
    SCAN_STATE["error"] = None
    try:
        CFG = load_cfg()
        result = SCAN.scan_directory(target, CFG, return_results=True)
        SCAN_STATE["result"] = result
    except Exception as e:
        SCAN_STATE["error"] = str(e)
    finally:
        SCAN_STATE["running"] = False


# ─── Ingest (run subprocess so it can be backgrounded) ──────────────────────

def ingest_active():
    """True if any ingest is running (spawned here, or externally via lock)."""
    return INGEST_STATE["running"] or ING.ingest_running()


def read_ingest_status():
    # Parse live log file if an ingest is running as a subprocess
    log_path = RAG_ROOT / "ingest.log"
    lines = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-400:]

    # Always keep the live log bytes in state — otherwise a freshly-truncated
    # log with no progress line yet leaves a *previous* run's lines on screen.
    INGEST_STATE["lines"] = lines[-80:]

    last = ""
    for ln in lines:
        if ln.startswith("[") and "/" in ln:
            last = ln.strip()
    # Parse progress from last line like "[12/500] done=10 skip=2 err=0 elapsed=1m0s eta=10m2s"
    if last:
        try:
            prefix = last.split("]")[0].strip("[").split("/")
            cur, total = int(prefix[0]), int(prefix[1])
            done = skipped = errors = 0
            for part in last.split():
                if part.startswith("done="): done = int(part.split("=")[1].rstrip(","))
                if part.startswith("skip="): skipped = int(part.split("=")[1].rstrip(","))
                if part.startswith("err="): errors = int(part.split("=")[1].rstrip(","))
            INGEST_STATE.update(
                current=cur, total=total, done=done, skipped=skipped, errors=errors,
                phase="stopping" if INGEST_STATE.get("stopping") else "indexing",
                message="Stopping \u2014 finishing current file then exiting\u2026"
                        if INGEST_STATE.get("stopping") else ""
            )
        except Exception:
            pass
    elif INGEST_STATE.get("running") or ING.ingest_running():
        # Running, but no per-file progress yet (embedder load / directory scan).
        stopping = INGEST_STATE.get("stopping")
        INGEST_STATE.update(current=0, total=0, done=0, skipped=0, errors=0,
                            phase="stopping" if stopping else "starting",
                            message="Stopping \u2014 finishing current file then exiting\u2026"
                                    if stopping else
                                    "Starting \u2014 loading embedding model / scanning directory\u2026")

    # Authoritative live PID: prefer the tracked Popen via poll() (reaps the
    # child and detects exit even with a zombie), else a live lock file, else
    # our tracked PID.
    global _INGEST_PROC
    ext = ING.read_ingest_lock()
    tracked = INGEST_STATE.get("pid")
    returncode = None
    live_pid = None
    if _INGEST_PROC is not None:
        returncode = _INGEST_PROC.poll()
        if returncode is None:
            live_pid = _INGEST_PROC.pid
    elif ext:
        p = int(ext.get("pid"))
        if _pid_alive(p):
            live_pid = p
            INGEST_STATE.update(set_name=ext.get("set"), target=ext.get("target"))
        elif _INGEST_PROC is None:
            _clean_stale_lock(RAG_ROOT / ".ingest.lock", ext)
    elif tracked and _pid_alive(tracked):
        live_pid = tracked

    if live_pid:
        # A job is running. Detect a newly-started run and clear stale flags.
        if INGEST_STATE.get("pid") != live_pid:
            INGEST_STATE.update(pid=live_pid, finished=False, completed=False,
                                paused=False)
        INGEST_STATE["running"] = True
    else:
        # No live ingest.
        was_running = INGEST_STATE["running"] or INGEST_STATE.get("pid") is not None
        INGEST_STATE["running"] = False
        INGEST_STATE["paused"] = False
        INGEST_STATE["pid"] = None
        INGEST_STATE["stopping"] = False
        INGEST_STATE["message"] = ""
        if was_running and not INGEST_STATE["finished"]:
            INGEST_STATE["finished"] = True
            INGEST_STATE["completed"] = (returncode == 0) if returncode is not None else True
            INGEST_STATE["phase"] = "done"
        elif not INGEST_STATE["finished"]:
            INGEST_STATE["phase"] = "idle"
        else:
            INGEST_STATE["phase"] = "done"
    return INGEST_STATE


def live_ingest_pid():
    """PID of a live ingest (spawned here or externally), else None."""
    if _INGEST_PROC is not None and _INGEST_PROC.poll() is None:
        return _INGEST_PROC.pid
    pid = INGEST_STATE.get("pid")
    if pid and _pid_alive(pid):
        return pid
    ext = ING.read_ingest_lock()
    if ext:
        p = int(ext.get("pid"))
        if _pid_alive(p):
            return p
    return None


def start_ingest(target, set_name, force=False, only=None):
    global _INGEST_PROC
    if ingest_active():
        return False, "An ingest is already running (started here or externally)."
    cfg = load_cfg()
    if not set_name:
        set_name = f"set{len(cfg['sets'])+1}"
    if not target:
        return False, "No source directory specified."
    # Remember this set/location so the Collections panel can offer update/rename.
    cfg.setdefault("sets", {})[set_name] = {"path": str(target), "kind": "local"}
    save_cfg(cfg)

    cmd = [sys.executable, str(SCRIPTS_DIR / "ingest.py"), str(target), "--set", set_name]
    if force:
        cmd.append("--force")
    if only:
        cmd += ["--only", only]

    INGEST_STATE.update(
        running=True, paused=False, finished=False, completed=False, total=0, current=0,
        done=0, skipped=0, errors=0, set_name=set_name, target=target,
        started_at=time.time(), lines=[], phase="starting",
        message="Starting \u2014 loading embedding model / scanning directory\u2026"
    )

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    log_path = RAG_ROOT / "ingest.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    _INGEST_PROC = proc
    INGEST_STATE["pid"] = proc.pid
    return True, f"Ingest started (PID {proc.pid})."


def _stop_watchdog(pid, state, lock_path, grace=8.0):
    """Escalate a graceful stop request.

    After SIGINT, the child finishes its current native call (embed/upsert) and
    removes its lock in a `finally`. But if it's stuck inside a long native call,
    Python won't run the SIGINT handler until it returns, so the UI would keep
    showing "indexing". This watchdog hard-kills after `grace` seconds and
    force-clears state + lock so chat frees up regardless.
    """
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return  # exited gracefully; its finally removed the lock
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    state.update(running=False, paused=False, pid=None, finished=True,
                 completed=False)
    try:
        lock_path.unlink()
    except OSError:
        pass


# ─── OCR (run subprocess, mirroring ingest) ─────────────────────────────────

def _read_ocr_lock():
    lock = RAG_ROOT / ".ocr.lock"
    try:
        return json.loads(lock.read_text())
    except Exception:
        return None


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _clean_stale_lock(lock_path, ext=None):
    """Remove a job lock whose recorded process is gone (crash/segfault/kill
    that prevented the child's `finally` from releasing it). A lingering lock
    would otherwise block new jobs and keep the UI wedged in "running"."""
    lock = lock_path or (RAG_ROOT / ".ocr.lock")
    try:
        if ext:
            pid = int(ext.get("pid"))
            if _pid_alive(pid):
                return  # genuinely live — leave it alone
        lock.unlink()
    except Exception:
        pass


def ocr_active():
    if OCR_STATE["running"] or OCR_STATE.get("pid"):
        return True
    if _OCR_PROC is not None and _OCR_PROC.poll() is None:
        return True
    ext = _read_ocr_lock()
    if not ext:
        return False
    if not _pid_alive(int(ext.get("pid"))):
        _clean_stale_lock(RAG_ROOT / ".ocr.lock", ext)
        return False
    return True


def read_ocr_status():
    log_path = RAG_ROOT / "ocr.log"
    lines = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-400:]

    # Always keep the live log bytes in state — a freshly-truncated log with no
    # progress line yet must not leave a previous run's lines on screen.
    OCR_STATE["lines"] = lines[-80:]

    last = ""
    for ln in lines:
        if ln.startswith("[") and "/" in ln:
            last = ln.strip()
    if last:
        try:
            prefix = last.split("]")[0].strip("[").split("/")
            cur, total = int(prefix[0]), int(prefix[1])
            done = skipped = errors = 0
            for part in last.split():
                if part.startswith("done="): done = int(part.split("=")[1].rstrip(","))
                if part.startswith("skip="): skipped = int(part.split("=")[1].rstrip(","))
                if part.startswith("err="): errors = int(part.split("=")[1].rstrip(","))
            OCR_STATE.update(current=cur, total=total, done=done,
                             skipped=skipped, errors=errors)
        except Exception:
            pass

    # Authoritative liveness: prefer the tracked Popen via poll() (which reaps
    # the child, so a finished-but-unreaped zombie exits cleanly), then a live
    # lock file, then our tracked PID.
    global _OCR_PROC
    ext = _read_ocr_lock()
    tracked = OCR_STATE.get("pid")
    returncode = None
    live_pid = None
    if _OCR_PROC is not None:
        returncode = _OCR_PROC.poll()
        if returncode is None:
            live_pid = _OCR_PROC.pid
    elif ext:
        p = int(ext.get("pid"))
        if _pid_alive(p):
            live_pid = p
        elif _OCR_PROC is None:
            _clean_stale_lock(RAG_ROOT / ".ocr.lock", ext)
    elif tracked and _pid_alive(tracked) and (ext is None):
        live_pid = tracked

    if live_pid:
        if OCR_STATE.get("pid") != live_pid:
            OCR_STATE.update(pid=live_pid, finished=False, completed=False, paused=False)
        if ext:
            OCR_STATE.update(target=ext.get("target"), mode=ext.get("mode"))
        OCR_STATE["running"] = True
    else:
        was_running = OCR_STATE["running"] or OCR_STATE.get("pid") is not None
        OCR_STATE["running"] = False
        OCR_STATE["paused"] = False
        OCR_STATE["pid"] = None
        if was_running and not OCR_STATE["finished"]:
            OCR_STATE["finished"] = True
            # Clean exit -> completed; crash/kill (non-zero/negative rc or a
            # leftover stale lock) -> failed so the UI can show the error state.
            OCR_STATE["completed"] = (returncode == 0) if returncode is not None else True
        if _OCR_PROC is not None and returncode is not None:
            _OCR_PROC = None
    return OCR_STATE


def live_ocr_pid():
    if _OCR_PROC is not None and _OCR_PROC.poll() is None:
        return _OCR_PROC.pid
    pid = OCR_STATE.get("pid")
    if pid and _pid_alive(pid):
        return pid
    ext = _read_ocr_lock()
    if ext:
        p = int(ext.get("pid"))
        if _pid_alive(p):
            return p
    return None


def start_ocr(target, mode="merge", force=False, only=None, languages=None, backend=None):
    global _OCR_PROC
    if ocr_active():
        return False, "An OCR job is already running."
    if not target:
        return False, "No source directory specified."
    cmd = [sys.executable, str(SCRIPTS_DIR / "ocr.py"), str(target), "--mode", mode]
    if force:
        cmd.append("--force")
    if only:
        cmd += ["--only", only]
    if languages:
        cmd += ["--languages", languages]
    if backend:
        cmd += ["--backend", backend]

    OCR_STATE.update(
        running=True, paused=False, finished=False, completed=False, total=0, current=0,
        done=0, skipped=0, errors=0, target=target, mode=mode, started_at=time.time()
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    log_path = RAG_ROOT / "ocr.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    _OCR_PROC = proc
    OCR_STATE["pid"] = proc.pid
    return True, f"OCR started (PID {proc.pid}, mode={mode})."


# ─── Chat helpers (reuse agent logic) ────────────────────────────────────────

_chat_client = None
_chat_embedder = None
_chat_lock = threading.Lock()


def get_chat_client():
    global _chat_client
    if _chat_client is None:
        cfg = load_cfg()
        _chat_client = __import__("agent", fromlist=["setup_client"]).setup_client(cfg)
    return _chat_client


def get_chat_collection(set_name):
    cfg = load_cfg()
    return __import__("agent", fromlist=["setup_chroma"]).setup_chroma(set_name)


def get_chat_embedder():
    global _chat_embedder
    if _chat_embedder is None:
        cfg = load_cfg()
        _chat_embedder = __import__("agent", fromlist=["setup_embedder"]).setup_embedder(cfg)
    return _chat_embedder


_reranker = {"model": None, "obj": None}


def get_chat_reranker():
    """Lazy-loaded cross-encoder reranker (None if disabled/config missing)."""
    global _reranker
    cfg = load_cfg()
    model = cfg.get("rerank_model")
    if not cfg.get("rerank_enabled", True) or not model:
        return None
    if _reranker["model"] != model:
        print(f"[chat] loading reranker: {model} ...", flush=True)
        try:
            from sentence_transformers import CrossEncoder
            _reranker["model"] = model
            _reranker["obj"] = CrossEncoder(model, device=cfg.get("embed_device", "cpu"))
        except Exception as e:
            print(f"[chat] reranker load failed, reranking disabled: {e}")
            _reranker["model"] = model
            _reranker["obj"] = None
    return _reranker["obj"]


# ─── Flask routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(RAG_ROOT / "web", "index.html")


@app.route("/web/<path:filename>")
def static_files(filename):
    return send_from_directory(RAG_ROOT / "web", filename)


@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify(_public_cfg(load_cfg()))


@app.route("/api/config", methods=["POST"])
def api_config_set():
    data = request.get_json(force=True)
    cfg = load_cfg()
    # Only allow whitelisted keys to update
    allowed = {
        "embed_model", "embed_device", "embed_dim", "chunk_tokens", "chunk_overlap",
        "chunking_strategy", "parent_tokens",
        "llm_base_url", "llm_model", "llm_api_key", "llm_temperature", "llm_max_tokens",
        "retrieval_top_k", "relevance_threshold", "max_retrieval_hops",
        "retrieval_hops_driver",
        "rerank_model", "rerank_enabled",
        "fiction_tags", "ocr_enabled", "ocr_languages",
        "ocr_merge", "ocr_backend",
        "ingest_batch_size",
    }
    for k in allowed:
        if k in data:
            cfg[k] = data[k]
    if "sets" in data:
        cfg["sets"] = data["sets"]
    save_cfg(cfg)
    return jsonify({"ok": True, "config": _public_cfg(cfg)})


@app.route("/api/system")
def api_system():
    llm = check_llm()
    return jsonify({
        "llm": llm,
        "lmstudio": llm,  # backward-compatible alias
        "source_dirs": detect_source_dirs(),
        "cuda": False,
        "drives": detect_source_dirs(),
    })


# ─── Curated embedding model picker ──────────────────────────────────────────

EMBED_MODELS = [
    {"id": "intfloat/multilingual-e5-small", "label": "multilingual-e5-small (CPU, multilingual)",
     "dim": 384, "est_mb": 130, "tier": "CPU / 3 GB", "mrl": False, "new": False},
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "label": "all-MiniLM-L6-v2 (fast, English)",
     "dim": 384, "est_mb": 90, "tier": "CPU / 3 GB", "mrl": False, "new": False},
    {"id": "BAAI/bge-small-en-v1.5", "label": "bge-small-en-v1.5 (English)",
     "dim": 384, "est_mb": 130, "tier": "CPU / 3 GB", "mrl": False, "new": False},
    {"id": "MongoDB/mdbr-leaf-mt", "label": "mdbr-leaf-mt (compact multilingual)",
     "dim": 768, "est_mb": 92, "tier": "CPU / 3 GB", "mrl": False, "new": True},
    {"id": "Qwen/Qwen3-Embedding-0.6B", "label": "Qwen3-Embedding-0.6B",
     "dim": 1024, "est_mb": 1300, "tier": "CPU / 3 GB", "mrl": True, "new": False},
    {"id": "BAAI/bge-m3", "label": "bge-m3 (multilingual, high quality)",
     "dim": 1024, "est_mb": 2300, "tier": "6 GB+ GPU", "mrl": True, "new": False},
    {"id": "google/embeddinggemma-300m", "label": "embeddinggemma-300m",
     "dim": 768, "est_mb": 1200, "tier": "6 GB+ GPU", "mrl": False, "new": True},
    {"id": "BidirLM/BidirLM-1.7B-Embedding", "label": "BidirLM-1.7B-Embedding",
     "dim": 1536, "est_mb": 3400, "tier": "6 GB+ GPU", "mrl": False, "new": True},
    {"id": "intfloat/multilingual-e5-base", "label": "multilingual-e5-base",
     "dim": 768, "est_mb": 1100, "tier": "6 GB+ GPU", "mrl": False, "new": False},
    {"id": "intfloat/multilingual-e5-large", "label": "multilingual-e5-large",
     "dim": 1024, "est_mb": 2300, "tier": "6 GB+ GPU", "mrl": True, "new": False},
]


def _hf_cache_dir_mb(model: str) -> int:
    """Best-effort size (MB) already on disk in the HF cache for a model."""
    try:
        import huggingface_hub
        snapshots = huggingface_hub.snapshot_download(model, local_files_only=True)
        total = 0
        for p in Path(snapshots).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return int(total // (1024 * 1024))
    except Exception:
        return 0


def _hf_est_size_mb(model: str) -> int | None:
    """Try to compute the true download size from the HF repo's file sizes."""
    try:
        import huggingface_hub
        sizes = {}
        for f in huggingface_hub.list_repo_files(model):
            info = huggingface_hub.hf_hub_download(model, f,
                                                   local_files_only=True)
            if info and Path(info).exists():
                sizes[f] = Path(info).stat().st_size
        if sizes:
            return int(sum(sizes.values()) // (1024 * 1024))
    except Exception:
        pass
    for e in EMBED_MODELS:
        if e["id"] == model:
            return e["est_mb"]
    return None


EMBED_LOAD_STATE = {
    "running": False,
    "model": None,
    "phase": "idle",
    "progress": 0,          # 0-100; -1 while unknown
    "est_mb": None,
    "real_mb": None,
    "dim": None,
    "done": False,
    "error": None,
}
EMBED_LOAD_LOCK = threading.Lock()


def _load_embed_model(model: str, device: str):
    """Background: download + instantiate a SentenceTransformer embedder."""
    est = _hf_est_size_mb(model)
    real_before = _hf_cache_dir_mb(model)
    with EMBED_LOAD_LOCK:
        EMBED_LOAD_STATE.update(running=True, model=model, phase="loading",
                                progress=0, est_mb=est, real_mb=real_before,
                                dim=None, done=False, error=None)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(os.cpu_count() or 8)
        # Monitor the HF cache while downloading (if not already cached).
        poll = threading.Event()

        def _watch():
            while not poll.wait(0.6):
                now = _hf_cache_dir_mb(model)
                with EMBED_LOAD_LOCK:
                    EMBED_LOAD_STATE["real_mb"] = now
                    if EMBED_LOAD_STATE["est_mb"]:
                        EMBED_LOAD_STATE["progress"] = min(
                            95, int(100 * now / EMBED_LOAD_STATE["est_mb"]))

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        try:
            embedder = SentenceTransformer(model, device=device or "cpu")
            dim = embedder.get_sentence_embedding_dimension()
        finally:
            poll.set()
        real = _hf_cache_dir_mb(model) or est or 0
        with EMBED_LOAD_LOCK:
            EMBED_LOAD_STATE.update(running=False, phase="done", progress=100,
                                    real_mb=real, dim=dim, done=True, error=None)
    except Exception as e:
        with EMBED_LOAD_LOCK:
            EMBED_LOAD_STATE.update(running=False, phase="error", progress=0,
                                    done=True, error=str(e))


@app.route("/api/embed/models")
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


@app.route("/api/embed/load", methods=["POST"])
def api_embed_load():
    data = request.get_json(force=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"ok": False, "error": "No model specified."}), 400
    with EMBED_LOAD_LOCK:
        if EMBED_LOAD_STATE["running"]:
            return jsonify({"ok": False,
                            "error": "A model load is already in progress."}), 409
    device = data.get("device") or "cpu"
    threading.Thread(target=_load_embed_model, args=(model, device),
                     daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/embed/load/status")
def api_embed_load_status():
    with EMBED_LOAD_LOCK:
        return jsonify(dict(EMBED_LOAD_STATE))


@app.route("/api/embed/apply", methods=["POST"])
def api_embed_apply():
    data = request.get_json(force=True) or {}
    cfg = load_cfg()
    model = (data.get("model") or "").strip()
    device = data.get("device") or cfg.get("embed_device", "cpu")
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
    global _chat_embedder
    _chat_embedder = None
    return jsonify({"ok": True, "config": cfg, "reindex": True})


# ─── LLM API configuration + dormant self-host plumbing ─────────────────────

def _find_llama_server():
    import shutil
    return shutil.which("llama-server")


SELFHOST_GGUF = "Qwen3-1.7B-Q4_K_M.gguf"
SELFHOST_REPO = "unsloth/Qwen3-1.7B-GGUF"
SELFHOST_URL = ("https://huggingface.co/" + SELFHOST_REPO +
                "/resolve/main/" + SELFHOST_GGUF)
SELFHOST_EST_MB = 1100  # ~1.1 GB


@app.route("/api/llm/status")
def api_llm_status():
    return jsonify(check_llm())


@app.route("/api/llm/apply", methods=["POST"])
def api_llm_apply():
    data = request.get_json(force=True) or {}
    cfg = load_cfg()
    if "llm_base_url" in data and data["llm_base_url"] is not None:
        cfg["llm_base_url"] = str(data["llm_base_url"]).strip()
    if "llm_model" in data and data["llm_model"] is not None:
        cfg["llm_model"] = str(data["llm_model"]).strip() or "default"
    if "llm_api_key" in data:
        key = str(data["llm_api_key"]).strip()
        if key and key != API_KEY_MASK:
            cfg["llm_api_key"] = key
        elif not key:
            cfg["llm_api_key"] = None
    save_cfg(cfg)
    global _chat_client
    _chat_client = None
    return jsonify({"ok": True, "config": _public_cfg(cfg)})


@app.route("/api/llm/selfhost")
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


SELFHOST_DL_STATE = {
    "running": False, "phase": "idle", "progress": 0,
    "est_mb": SELFHOST_EST_MB, "downloaded_mb": 0, "error": None,
}
SELFHOST_PROC = {"pid": None}


@app.route("/api/llm/download", methods=["POST"])
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


@app.route("/api/llm/download/status")
def api_llm_download_status():
    return jsonify(SELFHOST_DL_STATE)


@app.route("/api/llm/start", methods=["POST"])
def api_llm_start():
    binary = _find_llama_server()
    if not binary:
        return jsonify({"ok": False,
                        "error": "llama-server not found on this machine."}), 400
    models_dir = RAG_ROOT / "models"
    gguflist = [f for f in models_dir.glob("*.gguf")]
    if not gguflist:
        return jsonify({"ok": False,
                        "error": "No GGUF model in models/ yet."}), 400
    gguf = gguflist[0]
    if SELFHOST_PROC["pid"]:
        try:
            os.kill(SELFHOST_PROC["pid"], 0)
            return jsonify({"ok": False, "error": "llama-server already running."}), 409
        except OSError:
            SELFHOST_PROC["pid"] = None
    log_path = RAG_ROOT / "llama-server.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen([binary, "-m", str(gguf), "-c", "8192"],
                                stdout=logf, stderr=subprocess.STDOUT)
    SELFHOST_PROC["pid"] = proc.pid
    return jsonify({"ok": True, "pid": proc.pid, "model": str(gguf)})


@app.route("/api/llm/control", methods=["POST"])
def api_llm_control():
    data = request.get_json(force=True) or {}
    action = data.get("action")
    pid = SELFHOST_PROC["pid"]
    if not pid:
        return jsonify({"ok": False, "error": "llama-server is not running."}), 404
    try:
        os.kill(pid, 0)
    except OSError:
        SELFHOST_PROC["pid"] = None
        return jsonify({"ok": False, "error": "Process already exited."}), 404
    if action == "stop":
        os.kill(pid, signal.SIGTERM)
        SELFHOST_PROC["pid"] = None
        return jsonify({"ok": True, "message": "llama-server stopped."})
    return jsonify({"ok": False, "error": "Unknown action."}), 400


# ─── First-run wizard ────────────────────────────────────────────────────────

@app.route("/api/setup/status")
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


@app.route("/api/setup/finish", methods=["POST"])
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
    global _chat_client, _chat_embedder
    _chat_client = None
    _chat_embedder = None
    return jsonify({"ok": True, "config": _public_cfg(cfg)})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json(force=True) or {}
    target = data.get("target", "")
    if not target:
        return jsonify({"error": "No target directory"}), 400
    if SCAN_STATE["running"]:
        return jsonify({"error": "Scan already running"}), 409
    threading.Thread(target=run_scan, args=(target,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/scan/status")
def api_scan_status():
    return jsonify(SCAN_STATE)


@app.route("/api/ingest", methods=["POST"])
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


@app.route("/api/ingest/status")
def api_ingest_status():
    return jsonify(read_ingest_status())


@app.route("/api/ingest/control", methods=["POST"])
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


@app.route("/api/ocr", methods=["POST"])
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


@app.route("/api/ocr/status")
def api_ocr_status():
    return jsonify(read_ocr_status())


@app.route("/api/ocr/control", methods=["POST"])
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


def default_set_name():
    """Best default collection name for chat when none is specified."""
    cfg = load_cfg()
    if cfg.get("sets"):
        return next(iter(cfg["sets"]))
    return "veracrypt1"


def _prepare_rag(set_name, query, top_k, filter_kind, history=None):
    """Run retrieval and build the upstream LLM message list.

    Returns a dict:
      {"error": msg}                        on failure
      {"answer": "...", "sources": []}      when nothing relevant was found
      {"messages": [...], "sources": [...], "fiction_only": bool}
    """
    if ingest_active():
        return {"error": "Indexing is in progress — chat is paused until it "
                         "finishes. Check the Ingest tab for progress."}
    cfg = load_cfg()
    try:
        collection = get_chat_collection(set_name)
    except SystemExit:
        return {"error": f"Collection '{set_name}' not found. Run ingestion first."}
    if collection.count() == 0:
        return {"error": f"Collection '{set_name}' is empty."}

    embedder = get_chat_embedder()
    agent_mod = __import__("agent", fromlist=[
        "retrieve_rag", "SYSTEM_PROMPT"])

    rag_result = agent_mod.retrieve_rag(
        set_name, query, top_k, filter_kind, cfg,
        embedder=embedder, collection=collection,
        client=get_chat_client(),
        reranker=get_chat_reranker(),
    )

    title_mode = rag_result["title_mode"]
    matched_titles = rag_result["matched_titles"]
    context = rag_result["message_context"]
    fiction_only = rag_result["fiction_only"]
    low_rel = rag_result["low_relevance"]
    low_reason = rag_result["relevance_reason"]
    sources = rag_result["sources"]

    # Specific-work mode: the query named real library titles, so ask for a
    # per-work summary instead of a general answered-from-context question.
    if title_mode:
        user_msg = (
            "SUMMARIZE EACH of the requested work(s). For every listed work, give "
            "a separate summary covering: what the book is about, its main argument, "
            "structure, and notable ideas. Quote generously from the excerpts and "
            "label claims [LIBRARY].\n\n"
            f"Requested work(s): {', '.join(matched_titles)}\n\n"
            f"Retrieved excerpts:\n{context}")
        if fiction_only:
            user_msg += "\n\nNOTE: these works are FICTION. Present them as fiction, not fact."
    else:
        user_msg = f"Question: {query}\n\nRetrieved context:\n{context}"
        if low_rel:
            user_msg += (
                "\n\nNOTE: Retrieval found NO strong match for this query in the library "
                f"(reason: {low_reason or 'low relevance'}). Say clearly that the library "
                "lacks direct coverage, answer from your own knowledge labeled [KNOWLEDGE], "
                "and propose 2-3 related search phrases the user could try in the library.")
        elif fiction_only:
            user_msg += "\n\nNOTE: ALL retrieved sources are FICTION. Do NOT present them as factual. State clearly that these are fiction works."

    messages = [{"role": "system", "content": agent_mod.SYSTEM_PROMPT}]
    if history:
        for m in history:
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})

    return {"messages": messages, "sources": sources,
            "fiction_only": fiction_only,
            "low_relevance": low_rel, "relevance_reason": low_reason}


def run_rag_chat(set_name, query, top_k=None, filter_kind=None, history=None):
    """Blocking RAG chat. Returns (payload, http_code)."""
    if top_k is None:
        top_k = load_cfg().get("retrieval_top_k", 10)

    cfg = load_cfg()
    if cfg.get("agentic_enabled", True):
        try:
            agent_loop_mod = __import__("agent_loop", fromlist=["run_agent_loop"])
            result = agent_loop_mod.run_agent_loop(
                set_name, query, history, cfg, get_chat_client(),
                top_k=top_k, filter_kind=filter_kind,
                embedder=get_chat_embedder(),
                collection=get_chat_collection(set_name),
                reranker=get_chat_reranker(),
            )
            if result.get("error"):
                return ({"error": result["error"], "sources": result.get("sources", [])}), 500
            if not result.get("answer"):
                return ({"error": "LLM returned no usable answer. Is the LLM API "
                                 f"running and reachable at {cfg.get('llm_base_url')}?",
                         "sources": []}), 500
            return {"answer": result["answer"], "sources": result.get("sources", []),
                    "fiction_only": result.get("fiction_only", False),
                    "low_relevance": result.get("low_relevance", False),
                    "relevance_reason": result.get("relevance_reason", "")}, 200
        except Exception as e:
            print(f"[chat] agentic loop failed ({e}); falling back to single-shot", flush=True)

    payload = _prepare_rag(set_name, query, top_k, filter_kind, history)
    if "error" in payload:
        return payload, 409 if "paused" in payload["error"] else 404
    if "answer" in payload and "messages" not in payload:
        return payload, 200

    client = get_chat_client()
    answer = ""
    last_err = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=cfg.get("llm_model", "default"),
                messages=payload["messages"],
                temperature=cfg.get("llm_temperature", 0.3),
                max_tokens=cfg.get("llm_max_tokens", 2048),
            )
            msg = response.choices[0].message
            answer = (msg.content or "").strip()
            # Some models (Qwen3-style thinking) put everything in reasoning_content
            if not answer and getattr(msg, "reasoning_content", None):
                answer = msg.reasoning_content.strip()
            if answer:
                break
            last_err = "model returned an empty response"
            print(f"[chat] empty response on attempt {attempt+1}, retrying...")
        except Exception as e:
            last_err = str(e)
            print(f"[chat] LLM error attempt {attempt+1}: {e}")
            if "Failed to load model" in str(e):
                break  # model isn't loadable; retrying won't help
            time.sleep(1)

    if not answer:
        return ({"error": f"LLM returned no usable answer. {last_err}. "
                          f"Is the LLM API running and reachable at "
                          f"{load_cfg().get('llm_base_url')}?",
                 "sources": []}), 500

    return {"answer": answer, "sources": payload["sources"],
            "fiction_only": payload.get("fiction_only", False),
            "low_relevance": payload.get("low_relevance", False),
            "relevance_reason": payload.get("relevance_reason", "")}, 200


# Keep the llm-chat history bounded so it stays well inside the context window.
HISTORY_MSG_LIMIT = 12
HISTORY_CHAR_BUDGET = 6000


def _conversation_history(conv_id) -> list[dict]:
    """Load prior user/assistant messages (oldest first) within size limits."""
    if not conv_id:
        return []
    conv = chat_store.get_conversation(conv_id)
    if not conv:
        return []
    msgs = [m for m in conv.get("messages", [])
            if m.get("role") in ("user", "assistant") and m.get("content")]
    msgs = msgs[-HISTORY_MSG_LIMIT:]
    history, used = [], 0
    for m in msgs:
        content = (m.get("content") or "")[:1200]
        used += len(content) + 40
        if used > HISTORY_CHAR_BUDGET:
            break
        history.append({"role": m["role"], "content": content})
    return history


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    set_name = data.get("set", "veracrypt1")
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No question"}), 400
    top_k = int(data.get("top_k", load_cfg().get("retrieval_top_k", 10)))
    filter_kind = data.get("filter_kind")
    conv_id = data.get("conversation_id")
    history = _conversation_history(conv_id)
    result, code = run_rag_chat(set_name, query, top_k, filter_kind, history)

    if conv_id and "error" not in result:
        chat_store.add_message(conv_id, "user", query)
        chat_store.add_message(conv_id, "assistant", result.get("answer", ""),
                               {"sources": result.get("sources", []),
                                "fiction_only": result.get("fiction_only", False),
                                "low_relevance": result.get("low_relevance", False),
                                "relevance_reason": result.get("relevance_reason", "")})
    return jsonify(result), code


def _persist_chat(conv_id, query, result):
    """Persist a user/assistant turn to the conversation store on success."""
    if not conv_id or "error" in result:
        return
    chat_store.add_message(conv_id, "user", query)
    chat_store.add_message(conv_id, "assistant", result.get("answer", ""),
                           {"sources": result.get("sources", []),
                            "fiction_only": result.get("fiction_only", False),
                            "low_relevance": result.get("low_relevance", False),
                            "relevance_reason": result.get("relevance_reason", "")})


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming RAG chat for the web UI. Single-shot (no agentic loop).

    SSE events: ``delta`` (raw text), ``done`` (final JSON incl. sources/meta),
    ``error`` (JSON with an ``error`` string). Aborting mid-stream skips
    persistence.
    """
    data = request.get_json(force=True) or {}
    set_name = data.get("set", "veracrypt1")
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No question"}), 400
    top_k = int(data.get("top_k", load_cfg().get("retrieval_top_k", 10)))
    filter_kind = data.get("filter_kind")
    conv_id = data.get("conversation_id")
    history = _conversation_history(conv_id)
    return Response(
        _stream_sse(set_name, query, top_k, filter_kind, history, conv_id),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no"},
    )


def _stream_sse(set_name, query, top_k, filter_kind, history, conv_id):
    import json as _json
    cfg = load_cfg()
    payload = _prepare_rag(set_name, query, top_k, filter_kind, history)
    if "error" in payload:
        yield "event: error\ndata: " + _json.dumps({"error": payload["error"]}) + "\n\n"
        return
    if "answer" in payload and "messages" not in payload:
        text = payload["answer"]
        yield "event: delta\ndata: " + _json.dumps(text) + "\n\n"
        done = {"answer": text, "sources": payload.get("sources", []),
                "fiction_only": payload.get("fiction_only", False),
                "low_relevance": payload.get("low_relevance", False),
                "relevance_reason": payload.get("relevance_reason", "")}
        yield "event: done\ndata: " + _json.dumps(done) + "\n\n"
        _persist_chat(conv_id, query, done)
        return

    client = get_chat_client()
    full = []
    try:
        stream = client.chat.completions.create(
            model=cfg.get("llm_model", "default"),
            messages=payload["messages"],
            temperature=cfg.get("llm_temperature", 0.3),
            max_tokens=cfg.get("llm_max_tokens", 2048),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            text = None
            if delta is not None:
                text = getattr(delta, "content", None)
                if not text:
                    text = getattr(delta, "reasoning_content", None)
            if text:
                full.append(text)
                yield "event: delta\ndata: " + _json.dumps(text) + "\n\n"
    except Exception as e:
        yield "event: error\ndata: " + _json.dumps({"error": str(e)}) + "\n\n"
        return

    answer = "".join(full).strip()
    if not answer:
        yield "event: error\ndata: " + _json.dumps(
            {"error": "LLM returned no usable answer. Is the LLM API running?"}) + "\n\n"
        return
    done = {"answer": answer, "sources": payload.get("sources", []),
            "fiction_only": payload.get("fiction_only", False),
            "low_relevance": payload.get("low_relevance", False),
            "relevance_reason": payload.get("relevance_reason", "")}
    yield "event: done\ndata: " + _json.dumps(done) + "\n\n"
    _persist_chat(conv_id, query, done)


# ── OpenAI-compatible endpoints (for external AI chat clients) ───────────────

@app.route("/v1/models")
def api_openai_models():
    cfg = load_cfg()
    model_id = cfg.get("llm_model", "default")
    return jsonify({
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "rag-library-agent"}],
    })


@app.route("/v1/chat/completions", methods=["POST"])
def api_openai_chat():
    data = request.get_json(force=True) or {}
    if ingest_active():
        return jsonify({"error": {"message": "Indexing is in progress — chat is paused.",
                                  "type": "server_error", "code": "ingest_busy"}}), 503

    messages = data.get("messages") or []
    user_msgs = [m for m in messages
                 if m.get("role") == "user" and isinstance(m.get("content"), str)
                 and m.get("content", "").strip()]
    if not user_msgs:
        return jsonify({"error": {"message": "No user message found.",
                                  "type": "invalid_request_error"}}), 400
    query = user_msgs[-1]["content"].strip()
    history = messages[:-1]

    cfg = load_cfg()
    set_name = data.get("collection") or data.get("set") or data.get("user") \
        or default_set_name()
    top_k = int(data.get("top_k", cfg.get("retrieval_top_k", 10)))
    filter_kind = data.get("filter_kind")
    model = data.get("model") or cfg.get("llm_model", "default")

    stream = bool(data.get("stream", False))
    if stream:
        return Response(
            _sse_wrap(set_name, query, history, top_k, filter_kind, model),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no",
                     "Access-Control-Allow-Origin": "*"},
        )

    payload = _prepare_rag(set_name, query, top_k, filter_kind, history)
    if "error" in payload:
        code = 503 if "paused" in payload["error"] else 404
        return jsonify({"error": {"message": payload["error"],
                                  "type": "server_error"}}), code
    if "answer" in payload and "messages" not in payload:
        return _openai_response(payload["answer"], model, payload["sources"])

    # Agentic loop (non-streaming) when enabled; otherwise single-shot.
    if cfg.get("agentic_enabled", True):
        try:
            agent_loop_mod = __import__("agent_loop", fromlist=["run_agent_loop"])
            result = agent_loop_mod.run_agent_loop(
                set_name, query, history, cfg, get_chat_client(),
                top_k=top_k, filter_kind=filter_kind,
                embedder=get_chat_embedder(),
                collection=get_chat_collection(set_name),
                reranker=get_chat_reranker(),
            )
            if result.get("error"):
                return jsonify({"error": {"message": result["error"],
                                          "type": "server_error"}}), 500
            if result.get("answer"):
                return _openai_response(result["answer"], model,
                                        result.get("sources", []))
        except Exception as e:
            print(f"[chat] agentic loop failed ({e}); using single-shot", flush=True)

    client = get_chat_client()
    answer = ""
    last_err = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=payload["messages"],
                temperature=cfg.get("llm_temperature", 0.3),
                max_tokens=cfg.get("llm_max_tokens", 2048),
            )
            msg = response.choices[0].message
            answer = (msg.content or "").strip()
            if not answer and getattr(msg, "reasoning_content", None):
                answer = msg.reasoning_content.strip()
            if answer:
                break
            last_err = "model returned an empty response"
        except Exception as e:
            last_err = str(e)
            if "Failed to load model" in str(e):
                break
            time.sleep(1)
    if not answer:
        return jsonify({"error": {"message": f"LLM returned no usable answer. {last_err}.",
                                  "type": "server_error"}}), 500

    return _openai_response(answer, model, payload["sources"])


def _openai_response(content, model, sources):
    return jsonify({
        "id": "chatcmpl-" + os.urandom(6).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "sources": sources,
    })


def _sse_wrap(set_name, query, history, top_k, filter_kind, model):
    import json as _json
    payload = _prepare_rag(set_name, query, top_k, filter_kind, history)
    if "error" in payload:
        yield "event: error\n"
        yield f"data: {_json.dumps({'error': payload['error']})}\n\n"
        yield "data: [DONE]\n\n"
        return
    if "answer" in payload and "messages" not in payload:
        text = payload["answer"]
        for word in text.split(" "):
            yield "data: " + _json.dumps({
                "choices": [{"delta": {"content": word + " "}}]}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    cfg = load_cfg()
    client = get_chat_client()
    stream = client.chat.completions.create(
        model=model,
        messages=payload["messages"],
        temperature=cfg.get("llm_temperature", 0.3),
        max_tokens=cfg.get("llm_max_tokens", 2048),
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        text = getattr(delta, "content", None) if delta else None
        if text:
            yield "data: " + _json.dumps({
                "choices": [{"delta": {"content": text}}]}) + "\n\n"
    yield "data: [DONE]\n\n"


# ── Conversation management (for external clients and the web UI) ────────────

@app.route("/api/conversations")
def api_convs_list():
    return jsonify(chat_store.list_conversations())


@app.route("/api/conversations", methods=["POST"])
def api_convs_create():
    data = request.get_json(force=True) or {}
    conv = chat_store.create_conversation(title=data.get("title"),
                                          set=data.get("set", ""))
    return jsonify(conv), 201


@app.route("/api/conversations/<cid>")
def api_convs_get(cid):
    conv = chat_store.get_conversation(cid)
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv)


@app.route("/api/conversations/<cid>", methods=["PATCH"])
def api_convs_update(cid):
    data = request.get_json(force=True) or {}
    conv = chat_store.update_conversation(cid, title=data.get("title"),
                                          set=data.get("set"))
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv)


@app.route("/api/conversations/<cid>", methods=["DELETE"])
def api_convs_delete(cid):
    if chat_store.delete_conversation(cid):
        return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/conversations/<cid>/clear", methods=["POST"])
def api_convs_clear(cid):
    conv = chat_store.clear_conversation(cid)
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv)


@app.route("/api/collections")
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


def _chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=str(RAG_ROOT / "index"))


def _rename_manifest_set(old: str, new: str):
    """Point every manifest row for `old` at `new` so a later re-ingest
    doesn't double-count already-indexed files."""
    db = RAG_ROOT / "manifest.db"
    if not db.exists():
        return
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE files SET set_name = ? WHERE set_name = ?", (new, old))
    conn.commit()
    conn.close()


def _drop_manifest_set(set_name: str):
    db = RAG_ROOT / "manifest.db"
    if not db.exists():
        return
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM files WHERE set_name = ?", (set_name,))
    conn.commit()
    conn.close()


@app.route("/api/sets")
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


@app.route("/api/sets", methods=["POST"])
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


@app.route("/api/sets/<name>", methods=["DELETE"])
def api_sets_delete(name):
    if ingest_active():
        return jsonify({"ok": False, "message": "Ingest is running — wait for it to finish."}), 409
    cfg = load_cfg()
    if name in (cfg.get("sets") or {}):
        del cfg["sets"][name]
        save_cfg(cfg)
    return jsonify({"ok": True})


@app.route("/api/collections/rename", methods=["POST"])
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


@app.route("/api/collections/<name>", methods=["DELETE"])
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


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True})


@app.route("/api/fs")
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
