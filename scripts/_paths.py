#!/usr/bin/env python3
"""Central data-root resolution for the RAG app.

The app derives its data directory (config.json, index/, manifest.db,
conversations/, logs, .venv, etc.) from the location of the source code:
``Path(__file__).resolve().parent.parent``.

For a symlinked deployment (code lives in the canonical repo, but data lives
elsewhere, e.g. /path/to/Data/RAG), set the ``RAG_ROOT`` environment variable
to the data directory and every script will read/write data there instead of
next to the code. If unset, the code-relative default is used (the normal
in-place install).
"""

import json
import os
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent

# Keys that must never be persisted to the git-tracked config.json or returned
# to the client. They live in the gitignored config.local.json instead.
SECRET_KEYS = ("llm_api_key",)


def rag_root() -> Path:
    override = os.environ.get("RAG_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return _CODE_ROOT


def load_local_overrides() -> dict:
    """Read machine-specific/secrets overrides from config.local.json (gitignored)."""
    p = rag_root() / "config.local.json"
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def merge_local_config(cfg: dict) -> dict:
    """Overlay config.local.json over cfg so consumers see merged settings."""
    for k, v in load_local_overrides().items():
        cfg[k] = v
    return cfg
