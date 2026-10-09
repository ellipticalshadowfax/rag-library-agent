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

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent

# Keys that must never be persisted to the git-tracked config.json or returned
# to the client. They live in the gitignored config.local.json instead.
SECRET_KEYS = ("llm_api_key",)

# Store ids (conversations, quiz specs, quizzes) are generated as
# uuid4().hex[:12]; only that shape may be joined into a filesystem path.
_RAW_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def safe_id(value) -> str:
    """Return a validated store id, or "" for anything else.

    Guards raw-id path joins: a crafted id containing ``/``, ``..`` or other
    path metacharacters must never reach ``Path / id`` (see chat_store._path,
    quiz_store._spec_path/_quiz_path). Generated ids are always 12 hex chars.
    """
    s = str(value or "")
    return s if _RAW_ID_RE.match(s) else ""



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


def atomic_write_json(path, data, indent: int = 2) -> None:
    """Write ``data`` as JSON via a temp file + ``os.replace``.

    A crash or full disk mid-write can otherwise truncate config.json or a
    conversation file to invalid JSON, losing all settings/history. The unique
    temp name keeps concurrent writers from clobbering each other's partial file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix="." + p.name + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ocr_cache_name(fpath) -> str:
    """Return a collision-safe OCR cache filename for a source file.

    Keying the OCR cache by filename stem alone (``<stem>.txt``) lets two
    different PDFs that share a stem (e.g. ``Chapter 1.pdf`` in two folders, or
    the same filename across sets) read each other's OCR text; in merge mode
    that text is then written into the wrong PDF as an invisible text layer.
    Salt the name with a short hash of the file's resolved path so each source
    owns its cache. Callers must use this helper everywhere they read or write
    an OCR cache file.
    """
    p = Path(fpath)
    try:
        key = str(p.resolve())
    except OSError:
        key = str(p)
    digest = hashlib.sha1(key.replace("\\", "/").encode("utf-8")).hexdigest()[:8]
    return f"{p.stem}-{digest}.txt"


def cuda_available() -> bool:
    """True if a usable CUDA device is present and torch is CUDA-capable.

    torch is imported lazily so that lightweight CLI scripts that never do
    embeddings (e.g. pure bookkeeping) don't pay the torch import cost.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


#: Current embedding device key shorthand -> allowed device string.
_DEVICE_ALIASES = {
    "mps": "mps",      # Apple Silicon (unused by the UI; kept for manual config)
    "gpu": "cuda",     # UI shorthand for 'GPU (CUDA)'
}


def resolve_device(preferred=None):
    """Resolve the embedding/rerank device for this machine.

    ``preferred`` is the config value for ``embed_device`` — one of
      "auto" (or "", None) -> auto-detect: cuda if usable, else cpu
      "cpu" / "cuda"       -> exact override (respects user's explicit choice)
      "gpu"                -> alias for "cuda"
      "mps"                -> Apple Silicon Metal (manual config only)

    Returns the exact device string to pass to sentence-transformers / torch.
    """
    key = (preferred or "auto").strip().lower()
    key = _DEVICE_ALIASES.get(key, key)

    if key in ("cuda", "mps"):
        # Explicit override. Respect real availability so an explicit "cuda"
        # never hands sentence-transformers a missing backend (which the model
        # loaders would otherwise swallow and drop embeddings entirely).
        if key == "cuda" and not cuda_available():
            return "cpu"
        return key

    # "cpu" is an explicit override — honour it.
    if key == "cpu":
        return "cpu"

    # Anything else ("auto", unknown, empty, "gpu" alias unset) -> auto-detect.
    return "cuda" if cuda_available() else "cpu"

