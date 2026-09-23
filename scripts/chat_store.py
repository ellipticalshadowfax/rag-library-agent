#!/usr/bin/env python3
"""chat_store.py - JSON-file persistence for chat conversations.

Each conversation is a single JSON file under <repo>/conversations/. The layout
is intentionally simple and inspectable so users can edit or delete chats by
hand. The directory is gitignored and never committed.
"""

import json
import os
import time
import uuid
from pathlib import Path

from _paths import rag_root

CONV_DIR = rag_root() / "conversations"


def _path(cid: str) -> Path:
    return CONV_DIR / f"{cid}.json"


def _now() -> int:
    return int(time.time())


def list_conversations() -> list[dict]:
    """Return conversation summaries (no messages), newest-updated first."""
    CONV_DIR.mkdir(exist_ok=True)
    out = []
    for p in CONV_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "id": d.get("id"),
                "title": d.get("title", "Untitled"),
                "set": d.get("set", ""),
                "created": d.get("created", 0),
                "updated": d.get("updated", 0),
                "message_count": len(d.get("messages", [])),
            })
        except Exception:
            continue
    out.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return out


def get_conversation(cid: str) -> dict | None:
    p = _path(cid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_conversation(title: str = None, set: str = "") -> dict:
    CONV_DIR.mkdir(exist_ok=True)
    cid = uuid.uuid4().hex[:12]
    now = _now()
    conv = {
        "id": cid,
        "title": title or "New chat",
        "set": set or "",
        "created": now,
        "updated": now,
        "messages": [],
    }
    _path(cid).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv


def delete_conversation(cid: str) -> bool:
    p = _path(cid)
    if p.exists():
        p.unlink()
        return True
    return False


def update_conversation(cid: str, title: str = None, set: str = None) -> dict | None:
    conv = get_conversation(cid)
    if not conv:
        return None
    if title is not None:
        conv["title"] = title
    if set is not None:
        conv["set"] = set
    conv["updated"] = _now()
    _path(cid).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv


def clear_conversation(cid: str) -> dict | None:
    """Remove all messages and reset the title (so it re-auto-titles)."""
    conv = get_conversation(cid)
    if not conv:
        return None
    conv["messages"] = []
    conv["title"] = "New chat"
    conv.pop("pending_catalog", None)
    conv["updated"] = _now()
    _path(cid).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv


def set_pending_catalog(cid: str, pending: dict | None) -> dict | None:
    """Persist (or clear) the pending catalog clarification state."""
    conv = get_conversation(cid)
    if not conv:
        return None
    if pending:
        conv["pending_catalog"] = pending
    else:
        conv.pop("pending_catalog", None)
    conv["updated"] = _now()
    _path(cid).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv


def get_pending_catalog(cid: str) -> dict | None:
    conv = get_conversation(cid)
    if not conv:
        return None
    return conv.get("pending_catalog") or None


def set_pending_quiz_spec(cid: str, pending: dict | None) -> dict | None:
    """Persist (or clear) the pending quiz-spec proposal for a conversation.

    Used by SESSION 6's conversational spec negotiation: a quiz request stores a
    spec proposal here, and follow-up messages apply deltas to it. Currently
    only stored/cleared (endpoints in SESSION 2 expose it) and is not yet
    consumed by the chat flow.
    """
    conv = get_conversation(cid)
    if not conv:
        return None
    if pending:
        conv["pending_quiz_spec"] = pending
    else:
        conv.pop("pending_quiz_spec", None)
    conv["updated"] = _now()
    _path(cid).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv


def get_pending_quiz_spec(cid: str) -> dict | None:
    conv = get_conversation(cid)
    if not conv:
        return None
    return conv.get("pending_quiz_spec") or None


def add_message(cid: str, role: str, content: str, meta: dict = None) -> dict | None:
    """Append a message. Auto-titles the conversation from its first user turn."""
    conv = get_conversation(cid)
    if not conv:
        return None
    conv.setdefault("messages", []).append({
        "role": role,
        "content": content,
        "meta": meta or {},
        "ts": _now(),
    })
    if role == "user" and conv.get("title") == "New chat":
        title = " ".join(content.strip().split())
        conv["title"] = (title[:42] + "…") if len(title) > 42 else title
    conv["updated"] = _now()
    _path(cid).write_text(json.dumps(conv, indent=2), encoding="utf-8")
    return conv