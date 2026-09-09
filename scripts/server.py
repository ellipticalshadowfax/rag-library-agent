#!/usr/bin/env python3
"""RAG Web App - Setup, Scan, Ingest, and Chat in one browser UI.

Run:  source .venv/bin/activate && python scripts/server.py  (defaults to port 5000)
Open: http://localhost:5000
"""

import json
import os
import signal
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

RAG_ROOT = Path(__file__).resolve().parent.parent
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


# ─── Config helpers ──────────────────────────────────────────────────────────

def load_cfg():
    return ING.load_config()


def save_cfg(cfg):
    p = RAG_ROOT / "config.json"
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)


def default_cfg():
    return {
        "embed_model": "intfloat/multilingual-e5-small",
        "embed_device": "cpu",
        "embed_dim": 384,
        "chunk_tokens": 330,
        "chunk_overlap": 60,
        "llm_base_url": "http://localhost:1234/v1",
        "llm_model": "default",
        "llm_temperature": 0.3,
        "llm_max_tokens": 2048,
        "retrieval_top_k": 10,
        "fiction_tags": ["Fiction", "Short Stories", "Literary"],
        "ocr_enabled": False,
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


def check_lmstudio():
    """Check if LMStudio local server is reachable on the configured port."""
    cfg = load_cfg()
    import urllib.request
    url = cfg["llm_base_url"].rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read().decode())
        models = [m.get("id", "") for m in data.get("data", [])]
        return {"online": True, "models": models, "url": cfg["llm_base_url"]}
    except Exception as e:
        return {"online": False, "models": [], "error": str(e), "url": cfg["llm_base_url"]}


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
                lines=lines[-80:]
            )
        except Exception:
            pass

    # Authoritative live PID: prefer a live lock file, else our tracked PID.
    ext = ING.read_ingest_lock()
    tracked = INGEST_STATE.get("pid")

    live_pid = None
    if ext:
        live_pid = int(ext.get("pid"))
        INGEST_STATE.update(set_name=ext.get("set"), target=ext.get("target"))
    elif tracked:
        try:
            os.kill(tracked, 0)
            live_pid = tracked
        except OSError:
            pass

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
        if was_running and not INGEST_STATE["finished"]:
            INGEST_STATE["finished"] = True
            INGEST_STATE["completed"] = True
    return INGEST_STATE


def live_ingest_pid():
    """PID of a live ingest (spawned here or externally), else None."""
    pid = INGEST_STATE.get("pid")
    if pid:
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            pass
    ext = ING.read_ingest_lock()
    if ext:
        p = int(ext.get("pid"))
        try:
            os.kill(p, 0)
            return p
        except OSError:
            pass
    return None


def start_ingest(target, set_name, force=False, only=None):
    if ingest_active():
        return False, "An ingest is already running (started here or externally)."
    cfg = load_cfg()
    if not set_name:
        set_name = f"set{len(cfg['sets'])+1}"
    if not target:
        return False, "No source directory specified."

    cmd = [sys.executable, str(SCRIPTS_DIR / "ingest.py"), str(target), "--set", set_name]
    if force:
        cmd.append("--force")
    if only:
        cmd += ["--only", only]

    INGEST_STATE.update(
        running=True, paused=False, finished=False, completed=False, total=0, current=0,
        done=0, skipped=0, errors=0, set_name=set_name, target=target,
        started_at=time.time()
    )

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    log_path = RAG_ROOT / "ingest.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    INGEST_STATE["pid"] = proc.pid
    return True, f"Ingest started (PID {proc.pid})."


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


# ─── Flask routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(RAG_ROOT / "web", "index.html")


@app.route("/web/<path:filename>")
def static_files(filename):
    return send_from_directory(RAG_ROOT / "web", filename)


@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify(load_cfg())


@app.route("/api/config", methods=["POST"])
def api_config_set():
    data = request.get_json(force=True)
    cfg = load_cfg()
    # Only allow whitelisted keys to update
    allowed = {
        "embed_model", "embed_device", "chunk_tokens", "chunk_overlap",
        "llm_base_url", "llm_model", "llm_temperature", "llm_max_tokens",
        "retrieval_top_k", "fiction_tags", "ocr_enabled", "ocr_languages",
        "ingest_batch_size",
    }
    for k in allowed:
        if k in data:
            cfg[k] = data[k]
    if "sets" in data:
        cfg["sets"] = data["sets"]
    save_cfg(cfg)
    return jsonify({"ok": True, "config": cfg})


@app.route("/api/system")
def api_system():
    return jsonify({
        "lmstudio": check_lmstudio(),
        "source_dirs": detect_source_dirs(),
        "cuda": False,
        "drives": detect_source_dirs(),
    })


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
            INGEST_STATE["paused"] = False
            return jsonify({"ok": True,
                            "message": "Stop signal sent \u2014 finishing current file then exiting."})
    except ProcessLookupError:
        return jsonify({"ok": False, "message": "Process already exited."}), 404
    except PermissionError:
        return jsonify({"ok": False, "message": "No permission to control the process."}), 403
    return jsonify({"ok": False, "message": "Unknown action."}), 400


# Bound the prompt size: LM Studio's default window is 8192 tokens and the
# Qwen tokenizer is ~3 tokens/word. cap scored hits by words so the total
# prompt stays well under any default context window.
CONTEXT_WORD_BUDGET = 1500
CHUNK_WORD_CAP = 240


def default_set_name():
    """Best default collection name for chat when none is specified."""
    cfg = load_cfg()
    if cfg.get("sets"):
        return next(iter(cfg["sets"]))
    return "library"


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

    top_k = min(top_k, 8)
    embedder = get_chat_embedder()
    agent_mod = __import__("agent", fromlist=["retrieve", "build_context", "SYSTEM_PROMPT"])
    hits = agent_mod.retrieve(query, embedder, collection, top_k=top_k,
                              filter_kind=filter_kind, cfg=cfg)

    if not hits:
        return {"answer": "No relevant documents found.", "sources": [],
                "fiction_only": False}

    # Keep the highest-scoring hits that fit the word budget
    kept, used = [], 0
    for h in hits:
        if used + len(h["document"].split()) > CONTEXT_WORD_BUDGET:
            break
        kept.append(h)
        used += len(h["document"].split())
    hits = kept

    fiction_hits = [h for h in hits if h["metadata"].get("kind") == "fiction"]
    nonfiction_hits = [h for h in hits if h["metadata"].get("kind") != "fiction"]
    context = agent_mod.build_context(hits, max_words=CHUNK_WORD_CAP)

    user_msg = f"Question: {query}\n\nRetrieved context:\n{context}"
    if fiction_hits and not nonfiction_hits:
        user_msg += "\n\nNOTE: ALL retrieved sources are FICTION. Do NOT present them as factual. State clearly that these are fiction works."

    messages = [{"role": "system", "content": agent_mod.SYSTEM_PROMPT}]
    if history:
        for m in history:
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})

    sources = []
    for h in hits:
        m = h["metadata"]
        sources.append({
            "title": m.get("title", "?"),
            "source": m.get("source", "?"),
            "kind": m.get("kind", "unknown"),
            "tags": m.get("tags", "[]"),
            "score": round(1 - h["distance"], 3),
            "snippet": h["document"][:300],
        })

    return {"messages": messages, "sources": sources,
            "fiction_only": bool(fiction_hits and not nonfiction_hits)}


def run_rag_chat(set_name, query, top_k=None, filter_kind=None, history=None):
    """Blocking RAG chat. Returns (payload, http_code)."""
    if top_k is None:
        top_k = load_cfg().get("retrieval_top_k", 10)
    payload = _prepare_rag(set_name, query, top_k, filter_kind, history)
    if "error" in payload:
        return payload, 409 if "paused" in payload["error"] else 404
    if "answer" in payload and "messages" not in payload:
        return payload, 200

    cfg = load_cfg()
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
                          f"Is LMStudio running with a model loaded?",
                 "sources": []}), 500

    return {"answer": answer, "sources": payload["sources"],
            "fiction_only": payload["fiction_only"]}, 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    set_name = data.get("set", "library")
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No question"}), 400
    top_k = int(data.get("top_k", load_cfg().get("retrieval_top_k", 10)))
    filter_kind = data.get("filter_kind")
    result, code = run_rag_chat(set_name, query, top_k, filter_kind)

    conv_id = data.get("conversation_id")
    if conv_id and "error" not in result:
        chat_store.add_message(conv_id, "user", query)
        chat_store.add_message(conv_id, "assistant", result.get("answer", ""),
                               {"sources": result.get("sources", []),
                                "fiction_only": result.get("fiction_only", False)})
    return jsonify(result), code


# ── OpenAI-compatible endpoints (for external AI chat clients) ───────────────

@app.route("/v1/models")
def api_openai_models():
    cfg = load_cfg()
    model_id = cfg.get("llm_model", "default")
    return jsonify({
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "rag-web-chat"}],
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

    cfg = load_cfg()
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
    out = []
    for col in client.list_collections():
        out.append({"name": col.name, "count": col.count()})
    return jsonify(out)


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
    print(f"  LLM:    {cfg.get('llm_base_url')}  -> {check_lmstudio()}")
    print(f"  Config: {RAG_ROOT / 'config.json'}")
    print(f"  Index:  {RAG_ROOT / 'index'}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
