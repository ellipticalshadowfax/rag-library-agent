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
import time
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent

# Keys that must never be persisted to the git-tracked config.json or returned
# to the client. They live in the gitignored config.local.json instead.
SECRET_KEYS = ("llm_api_key",)

# Machine-specific keys (absolute local paths) that live in config.local.json
# but are returned to the client like any other setting. Keeping them out of
# config.json stops every collection save from dirtying the tracked file.
LOCAL_ONLY_KEYS = ("sets",)

# Everything that is written to config.local.json rather than config.json.
LOCAL_KEYS = SECRET_KEYS + LOCAL_ONLY_KEYS

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


def migrate_local_only_keys(cfg_path=None) -> None:
    """One-time move of LOCAL_ONLY_KEYS from config.json to config.local.json.

    Existing installs keep ``sets`` inside the tracked config.json. For each
    local-only key found there, the value is copied into config.local.json
    unless that file already sets it, then the key is removed from config.json.
    The effective merged config is unchanged by this: when both files carry the
    key, config.local.json already wins the merge, so dropping the config.json
    copy changes nothing a reader sees. Local is written before the tracked
    file, so a crash between the two leaves a duplicate, never a loss. Failures
    (read-only checkout, bad JSON) are reported and otherwise ignored, because
    the merged view still works without the migration.
    """
    cfg_path = Path(cfg_path) if cfg_path else rag_root() / "config.json"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    found = [k for k in LOCAL_ONLY_KEYS if k in raw]
    if not found:
        return
    try:
        local = load_local_overrides()
        changed = False
        for k in found:
            if k not in local:
                local[k] = raw[k]
                changed = True
        if changed:
            atomic_write_json(cfg_path.parent / "config.local.json", local)
        for k in found:
            raw.pop(k, None)
        atomic_write_json(cfg_path, raw)
    except OSError as e:
        import sys
        print(f"[config] could not migrate {found} out of config.json: {e}",
              file=sys.stderr)


def load_config_file() -> dict:
    """Load config.json with local-only keys migrated, merged with local overrides.

    Every reader of the app config should go through this, not json.load on
    config.json directly, so none of them miss keys that now live locally.
    """
    cfg_path = rag_root() / "config.json"
    migrate_local_only_keys(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return merge_local_config(cfg)


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


#: Per-model query/passage prefixes: lowercase substring of the model name ->
#: (query_prefix, passage_prefix). First match wins. Overridable per install via
#: config ``embed_query_prefix`` / ``embed_passage_prefix``.
EMBED_PREFIX_TABLE = [
    ("e5", ("query: ", "passage: ")),
    ("qwen3-embedding", (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: ", "")),
]


def resolve_embed_prefixes(cfg, model):
    """Return ``(query_prefix, passage_prefix)`` for ``model``.

    Config keys ``embed_query_prefix`` / ``embed_passage_prefix`` win when
    present and not None (an explicit ``""`` means "no prefix"); otherwise the
    model-name table applies, and unknown models get ``""``.
    """
    name = (model or "").lower()
    tq, tp = "", ""
    for needle, (q, p) in EMBED_PREFIX_TABLE:
        if needle in name:
            tq, tp = q, p
            break
    q = cfg.get("embed_query_prefix")
    p = cfg.get("embed_passage_prefix")
    return (tq if q is None else str(q)), (tp if p is None else str(p))


def embedder_identity(cfg, model=None):
    """Identity of the configured embedder, for collection metadata / mismatch
    checks: ``{backend, model, dim, query_prefix, passage_prefix}``.

    Pass the live ``model`` to use its real values (remote embedders report the
    probed dim; sentence-transformers models report their ``prompts``). Without
    it, dim comes from ``cfg['embed_dim']`` and prefixes from config/table.
    """
    ident = getattr(model, "identity", None)
    if isinstance(ident, dict):
        return dict(ident)
    backend = (cfg.get("embed_backend") or "local").strip().lower()
    name = cfg.get("embed_model", "intfloat/multilingual-e5-small")
    q, p = resolve_embed_prefixes(cfg, name)
    prompts = getattr(model, "prompts", None)
    if isinstance(prompts, dict):
        q = prompts.get("query", q)
        p = prompts.get("passage", p)
    dim = cfg.get("embed_dim")
    try:
        dim = int(dim) if dim is not None else None
    except (TypeError, ValueError):
        dim = None
    return {"backend": backend, "model": name, "dim": dim,
            "query_prefix": q, "passage_prefix": p}


def _default_sleep(seconds):
    time.sleep(seconds)


def _default_clock():
    return time.monotonic()


class OllamaRemoteEmbedder:
    """Drop-in stand-in for a sentence-transformers model that calls a remote
    Ollama ``/api/embed`` endpoint instead of loading weights in-process.

    Selected via config ``embed_backend: "ollama"``. Lets a heavy/bulk index
    build run against a GPU box's Ollama service while query-time code paths
    keep calling the same ``.encode()`` interface unchanged.

    ``encode(prompt_name="query"|"passage")`` prepends the matching prefix;
    any other prompt_name is sent unprefixed. Transient failures
    (ConnectionError/Timeout/HTTP 5xx) are retried twice with 0.5s/1s backoff;
    4xx is never retried. ``sleep`` is injectable for tests.
    """

    RETRY_BACKOFF = (0.5, 1.0)

    def __init__(self, base_url: str, model: str, dim: int = 1024, timeout: int = 180,
                 query_prefix: str = "", passage_prefix: str = "", sleep=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dim
        self._probed_dim = None
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._sleep = sleep
        # Real prefixes, so callers' ``"query" not in model.prompts`` guards
        # do not overwrite them.
        self.prompts = {"query": query_prefix, "passage": passage_prefix}

    def get_sentence_embedding_dimension(self):
        """Real dim, probed lazily with a tiny encode and cached; falls back to
        the configured dim if the endpoint is unreachable (not cached)."""
        if self._probed_dim is None:
            try:
                vec = self.encode("dimension probe")
                self._probed_dim = int(len(vec))
            except Exception:
                return self._dim
        return self._probed_dim

    def get_embedding_dimension(self):  # newer sentence-transformers alias
        return self.get_sentence_embedding_dimension()

    @property
    def identity(self):
        return {"backend": "ollama", "model": self.model,
                "dim": self.get_sentence_embedding_dimension(),
                "query_prefix": self.query_prefix,
                "passage_prefix": self.passage_prefix}

    def _post(self, batch):
        import requests

        delays = list(self.RETRY_BACKOFF)
        while True:
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                transient = (not isinstance(e, requests.HTTPError)) or (
                    status is not None and status >= 500)
                if not transient or not delays:
                    raise
                (self._sleep or _default_sleep)(delays.pop(0))

    def encode(self, texts, convert_to_numpy=True, batch_size=64,
               show_progress_bar=False, prompt_name=None, **kwargs):
        import numpy as np

        prefix = {"query": self.query_prefix,
                  "passage": self.passage_prefix}.get(prompt_name, "")
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if prefix:
            items = [prefix + t for t in items]
        out = []
        for i in range(0, len(items), max(1, batch_size)):
            data = self._post(items[i:i + batch_size])
            out.extend(data["embeddings"])
        arr = np.array(out, dtype="float32")
        return arr[0] if single else arr


def build_embedder(cfg):
    """Factory for the embedding model: local sentence-transformers (default)
    or a remote Ollama backend when ``cfg['embed_backend'] == "ollama"``.

    Centralizing this means ingest.py / agent.py / eval.py / server_core.py
    all pick up the remote path by switching one config key.
    """
    backend = (cfg.get("embed_backend") or "local").strip().lower()
    if backend == "ollama":
        base_url = cfg.get("embed_api_url", "http://192.168.88.129:11434")
        model = cfg.get("embed_model", "qwen3-embedding:0.6b")
        dim = int(cfg.get("embed_dim", 1024))
        qp, pp = resolve_embed_prefixes(cfg, model)
        return OllamaRemoteEmbedder(
            base_url, model, dim=dim,
            timeout=int(cfg.get("embed_timeout", 180)),
            query_prefix=qp, passage_prefix=pp)

    import torch
    from sentence_transformers import SentenceTransformer
    model_name = cfg.get("embed_model", "intfloat/multilingual-e5-small")
    dev = resolve_device(cfg.get("embed_device"))
    torch.set_num_threads(os.cpu_count() or 8)
    model = SentenceTransformer(model_name, device=dev)
    return model


class RemoteReranker:
    """Drop-in stand-in for sentence-transformers' CrossEncoder that calls a
    remote llama-server ``--reranking`` endpoint (``/v1/rerank``).

    Ollama has no rerank endpoint (ollama/ollama#3368, long-standing); a
    separate llama-server process is the legitimate way to serve a
    cross-encoder's real classification-head score remotely instead of the
    embedding-magnitude workaround some projects use. Selected via config
    ``rerank_backend: "remote"``.

    Scores are backend-scaled (raw logits vs. 0-1 probabilities) and are NOT
    comparable across backends or models; never threshold on them.

    Circuit breaker: after a failed call, ``predict`` raises immediately for
    ``cooldown`` seconds instead of touching the network; a success resets it.
    ``clock`` is injectable for tests.
    """

    def __init__(self, base_url: str, model: str = "reranker", timeout: int = 15,
                 cooldown: float = 30, clock=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cooldown = cooldown
        self._clock = clock
        self._open_until = None

    def _now(self):
        return (self._clock or _default_clock)()

    def predict(self, pairs, **kwargs):
        import requests

        if not pairs:
            return []
        if self._open_until is not None and self._now() < self._open_until:
            raise RuntimeError("remote reranker unavailable (circuit open)")
        query = pairs[0][0]
        docs = [p[1] for p in pairs]
        try:
            resp = requests.post(
                f"{self.base_url}/v1/rerank",
                json={"model": self.model, "query": query, "documents": docs,
                      "top_n": len(docs)},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            self._open_until = self._now() + self.cooldown
            raise
        self._open_until = None
        results = data.get("results") or []
        if not results:
            raise ValueError("remote reranker returned no results")
        returned = {r["index"]: r["relevance_score"] for r in results}
        floor = min(returned.values()) - 1.0
        return [returned.get(i, floor) for i in range(len(docs))]


def build_reranker(cfg):
    """Factory for the reranker: local sentence-transformers CrossEncoder
    (default) or a remote llama-server backend when
    ``cfg['rerank_backend'] == "remote"``."""
    backend = (cfg.get("rerank_backend") or "local").strip().lower()
    if backend == "remote":
        base_url = cfg.get("rerank_api_url", "http://192.168.88.129:11435")
        model = cfg.get("rerank_model", "qwen3-reranker")
        return RemoteReranker(
            base_url, model,
            timeout=cfg.get("rerank_timeout", 15),
            cooldown=cfg.get("rerank_cooldown", 30))
    from sentence_transformers import CrossEncoder
    model = cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder(model, device=resolve_device(cfg.get("embed_device")))


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

