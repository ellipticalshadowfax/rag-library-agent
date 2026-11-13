"""Chat, streaming chat, OpenAI-compatible and conversation routes.

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
    _as_int,
    _conversation_history,
    _openai_response,
    _prepare_rag,
    _resolve_pending_spec_delta,
    _sse_wrap,
    _stream_sse,
    _turn_meta,
    default_set_name,
    get_chat_client,
    get_chat_collection,
    get_chat_embedder,
    get_chat_reranker,
    ingest_active,
    load_cfg,
    maybe_catalog,
    maybe_quiz,
    run_rag_chat,
)

bp = Blueprint("chat", __name__)


@bp.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    set_name = data.get("set", "library")
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No question"}), 400
    top_k = _as_int(data.get("top_k", load_cfg().get("retrieval_top_k", 10)),
                    10, 1, 50)
    filter_kind = data.get("filter_kind")
    conv_id = data.get("conversation_id")
    quiz = maybe_quiz(set_name, query, conv_id, filter_kind)
    if quiz is not None:
        if conv_id and "error" not in quiz:
            chat_store.add_message(conv_id, "user", query)
            chat_store.add_message(conv_id, "assistant", quiz.get("answer", ""),
                                   _turn_meta(quiz))
        return jsonify(quiz), 200
    delta = _resolve_pending_spec_delta(conv_id, query)
    if delta is not None:
        if conv_id and "error" not in delta:
            chat_store.add_message(conv_id, "user", query)
            chat_store.add_message(conv_id, "assistant", delta.get("answer", ""),
                                   _turn_meta(delta))
        return jsonify(delta), 200
    cat = maybe_catalog(set_name, query, conv_id, filter_kind)
    if cat is not None:
        if conv_id and "error" not in cat:
            chat_store.add_message(conv_id, "user", query)
            chat_store.add_message(conv_id, "assistant", cat.get("answer", ""),
                                   _turn_meta(cat))
        return jsonify(cat), 200
    history = _conversation_history(conv_id)
    result, code = run_rag_chat(set_name, query, top_k, filter_kind, history)

    if conv_id and "error" not in result:
        chat_store.add_message(conv_id, "user", query)
        chat_store.add_message(conv_id, "assistant", result.get("answer", ""),
                               _turn_meta(result))
    return jsonify(result), code


@bp.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming RAG chat for the web UI.

    SSE events: ``delta`` (raw text), ``done`` (final JSON incl. sources/meta),
    ``progress`` (optional heartbeat), ``error`` (JSON with an ``error``
    string). Aborting mid-stream skips persistence. Under ``chat_mode:
    "agentic"`` this runs the streaming agentic loop (progress heartbeats +
    streamed final answer); otherwise it stays single-shot.
    """
    data = request.get_json(force=True) or {}
    set_name = data.get("set", "library")
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No question"}), 400
    top_k = _as_int(data.get("top_k", load_cfg().get("retrieval_top_k", 10)),
                    10, 1, 50)
    filter_kind = data.get("filter_kind")
    conv_id = data.get("conversation_id")
    history = _conversation_history(conv_id)
    return Response(
        _stream_sse(set_name, query, top_k, filter_kind, history, conv_id),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no"},
    )


# ── OpenAI-compatible endpoints (for external AI chat clients) ───────────────

@bp.route("/v1/models")
def api_openai_models():
    cfg = load_cfg()
    model_id = cfg.get("llm_model", "default")
    return jsonify({
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "rag-library-agent"}],
    })


@bp.route("/v1/chat/completions", methods=["POST"])
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
    top_k = _as_int(data.get("top_k", cfg.get("retrieval_top_k", 10)), 10, 1, 50)
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

    quiz = maybe_quiz(set_name, query, None, filter_kind)
    if quiz is not None:
        return _openai_response(quiz["answer"], model, [])
    cat = maybe_catalog(set_name, query, None, filter_kind)
    if cat is not None:
        return _openai_response(cat["answer"], model, [])

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


# ── Conversation management (for external clients and the web UI) ────────────

@bp.route("/api/conversations")
def api_convs_list():
    return jsonify(chat_store.list_conversations())


@bp.route("/api/conversations", methods=["POST"])
def api_convs_create():
    data = request.get_json(force=True) or {}
    conv = chat_store.create_conversation(title=data.get("title"),
                                          set=data.get("set", ""))
    return jsonify(conv), 201


@bp.route("/api/conversations/<cid>")
def api_convs_get(cid):
    conv = chat_store.get_conversation(cid)
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv)


@bp.route("/api/conversations/<cid>", methods=["PATCH"])
def api_convs_update(cid):
    data = request.get_json(force=True) or {}
    conv = chat_store.update_conversation(cid, title=data.get("title"),
                                          set=data.get("set"))
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv)


@bp.route("/api/conversations/<cid>", methods=["DELETE"])
def api_convs_delete(cid):
    if chat_store.delete_conversation(cid):
        return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


@bp.route("/api/conversations/<cid>/clear", methods=["POST"])
def api_convs_clear(cid):
    conv = chat_store.clear_conversation(cid)
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv)
