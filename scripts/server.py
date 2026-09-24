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
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Skip HuggingFace remote checks; models cached from first-run

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

from _paths import SECRET_KEYS, rag_root

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
        "max_tokens_default": 2048,
        "max_tokens_quiz": 4096,
        "max_tokens_summary": 4096,
        "chat_mode": "agentic",
        "show_thinking": False,
        "setup_complete": False,
        "context_word_budget": 3000,
        "chunk_word_cap": 300,
        "history_char_budget": 10000,
        "history_msg_limit": 12,
        "catalog_enabled": True,
        "catalog_clarify_threshold": 20,
        "catalog_max_rows": 200,
        "catalog_semantic_pool": 500,
        "summary_strategy": "map_reduce",
        "summary_max_chunks": 40,
        "summary_context_budget": 8000,
        "quiz_default_count": 10,
        "quiz_output_format": "markdown",
        # Quiz planning knobs (SESSION 2).
        "quiz_depth": "balanced",
        "quiz_topic_max_works": 3,
        "quiz_topic_section_pool": 120,
        # Quiz material caps (SESSION 1: consolidated from server.py/agent_loop.py
        # /mcp_server.py so all three sites read the same config keys).
        "quiz_material_chunks": 10,
        "quiz_material_words": 500,
        # Quiz generation + audit pipeline (SESSION 3): per-batch generation over
        # the parents walk, the 4-stage accuracy audit, and the structured-output
        # capability probe. Advanced keys, tool-tweakable in config.json.
        "quiz_batch_parents": 4,
        "quiz_parent_word_cap": 500,
         "quiz_max_batches": 12,
        # Fallback chunk sampling for non-parent_child indexes (books without
        # parent-child sections in manifest.db). Paginates Chroma directly to
        # get evenly-spaced chunks so the LLM sees broad coverage, not just a
        # handful from semantic search top-k.
        "quiz_sample_children": 50,
        "quiz_verify_pass": True,
        "quiz_grounding_ratio": 0.85,
        "quiz_dedupe_jaccard": 0.75,
        "llm_structured_output": "auto",
        # Quiz grading + analytics (SESSION 4): deterministic grading for
        # mcq/true_false/fill_blank, LLM-assisted grading for short answers, and
        # the weak unit threshold for analytics. Advanced keys.
        "quiz_grade_llm": True,
        "quiz_weak_threshold": 0.6,
        # FSRS spaced-repetition review (SESSION 5): when true, missed/weak
        # questions are enrolled into the shared review store at attempt
        # finish so they resurface on an FSRS schedule in the Review pane.
        "fsrs_enabled": True,
        "quiz_review_limit": 50,
        "retrieval_top_k": 10,
        "relevance_threshold": 0.80,
        "max_retrieval_hops": 2,
        "retrieval_hops_driver": "llm",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_enabled": True,
        # Advanced/internal retrieval tuning (SESSION 1: promoted from hardcoded
        # constants in agent.py so any tool can read/tweak them in one place.
        # NOT rendered in the Setup tab; defaults keep retrieval bit-identical).
        "fuse_rrf_k": 60,
        "bm25_top_n": 30,
        "bm25_batch": 20000,
        "bm25_min_coverage": 0.9,
        "rerank_top_n": 24,
        "max_chunks_per_title": 3,
        "dense_pool_multiplier": 3,
        "fused_pool_floor": 40,
        "lexical_backend": "auto",
        "fiction_tags": ["Fiction", "Short Stories", "Literary"],
        "ocr_enabled": False,
        "ocr_merge": True,
        "ocr_backend": "rapidocr",
        "ocr_char_threshold": 50,
        "ocr_languages": ["en"],
        "sets": {},
    }


def cfg_int(cfg: dict, key: str, default: int, lo=None, hi=None) -> int:
    """Read an int config key, clamped to its registry [lo, hi] range.

    Falls back to ``default`` when the key is missing/invalid, matching the
    module-level constant default so retrieval behavior stays bit-identical
    unless a key is explicitly set in config.json.
    """
    try:
        v = int(cfg.get(key, default))
    except (TypeError, ValueError):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def cfg_float(cfg: dict, key: str, default: float, lo=None, hi=None) -> float:
    """Read a float config key, clamped to its registry [lo, hi] range."""
    try:
        v = float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


# ─── Config registry (single source of metadata) ─────────────────────────────
#
# `default_cfg()` above is the single source of DEFAULTS; CONFIG_META here is
# the single source of METADATA (label / group / range / advanced flag /
# description) for every config key. Never hand-maintain the POST allowlist —
# it is derived from this registry (see api_config_set). Keys with
# ``editable=False`` are managed values (setup state, per-set data) that the
# API must not overwrite. Advanced keys are tool-tweakable (settable in
# config.json) but are NOT shown as UI controls in the Setup tab.
CONFIG_META = {
    # Model / embedding
    "embed_model":       {"label": "Embed model", "group": "Model", "advanced": False, "editable": True, "description": "Sentence-transformers embedding model id."},
    "embed_device":      {"label": "Embed device", "group": "Model", "advanced": False, "editable": True, "description": "Device used for embeddings (cpu)."},
    "embed_dim":         {"label": "Embed dim", "group": "Model", "advanced": False, "editable": True, "description": "Embedding dimensionality for the collection."},

    # Indexing
    "chunk_tokens":      {"label": "Chunk tokens", "group": "Indexing", "min": 64, "max": 2000, "advanced": False, "editable": True, "description": "Target tokens per chunk for flat indexing."},
    "chunk_overlap":     {"label": "Chunk overlap", "group": "Indexing", "min": 0, "max": 500, "advanced": False, "editable": True, "description": "Token overlap between adjacent chunks."},
    "chunking_strategy": {"label": "Chunking strategy", "group": "Indexing", "advanced": False, "editable": True, "description": "Indexing granularity: flat or parent_child."},
    "parent_tokens":     {"label": "Parent tokens", "group": "Indexing", "min": 100, "advanced": False, "editable": True, "description": "Section size for parent_child generation."},
    "ingest_batch_size": {"label": "Ingest batch size", "group": "Indexing", "min": 1, "max": 5000, "advanced": False, "editable": True, "description": "Chroma upsert/embed batch size."},

    # Agentic
    "agentic_enabled":   {"label": "Agentic enabled", "group": "Chat", "advanced": True, "editable": False, "description": "Enable the tool-calling agent loop (managed)."},
    "agentic_max_steps": {"label": "Agentic max steps", "group": "Chat", "min": 1, "max": 20, "advanced": True, "editable": False, "description": "Max tool-calling steps in the agent loop (managed)."},
    "agentic_strategy":  {"label": "Agentic strategy", "group": "Chat", "advanced": True, "editable": False, "description": "Agentic loop strategy (auto)."},

    # LLM
    "llm_base_url":      {"label": "LLM base URL", "group": "LLM", "advanced": False, "editable": True, "description": "Base URL of the OpenAI-compatible server."},
    "llm_model":         {"label": "LLM model", "group": "LLM", "advanced": False, "editable": True, "description": "Model name served by the LLM server."},
    "llm_api_key":       {"label": "LLM API key", "group": "LLM", "advanced": False, "editable": True, "description": "Optional API key (masked; stored in config.local.json)."},
    "llm_temperature":   {"label": "LLM temperature", "group": "LLM", "min": 0, "max": 2, "advanced": False, "editable": True, "description": "Generation temperature."},
    "llm_max_tokens":    {"label": "LLM max tokens", "group": "LLM", "min": 64, "advanced": False, "editable": True, "description": "Default max tokens for generation."},
    "max_tokens_default": {"label": "Max tokens (default QA)", "group": "LLM", "min": 64, "advanced": False, "editable": True, "description": "Max tokens for normal QA."},
    "max_tokens_quiz":   {"label": "Max tokens (quiz)", "group": "LLM", "min": 64, "advanced": False, "editable": True, "description": "Max tokens for quiz generation."},
    "max_tokens_summary": {"label": "Max tokens (summary)", "group": "LLM", "min": 64, "advanced": False, "editable": True, "description": "Max tokens for summaries."},

    # Chat
    "chat_mode":         {"label": "Chat mode", "group": "Chat", "advanced": False, "editable": True, "description": "agentic (tool loop) or single (single-shot)."},
    "show_thinking":     {"label": "Show model thinking", "group": "Chat", "advanced": False, "editable": True, "description": "Stream reasoning_content when available."},
    "setup_complete":    {"label": "Setup complete", "group": "Chat", "advanced": True, "editable": False, "description": "Whether first-run setup has been completed (managed)."},
    "context_word_budget": {"label": "Context word budget", "group": "Chat", "min": 200, "max": 20000, "advanced": False, "editable": True, "description": "Total words of retrieved context sent to the LLM."},
    "chunk_word_cap":    {"label": "Chunk word cap", "group": "Chat", "min": 50, "max": 2000, "advanced": False, "editable": True, "description": "Per-chunk word truncation for flat/child chunks."},
    "history_char_budget": {"label": "History char budget", "group": "Chat", "min": 0, "max": 100000, "advanced": False, "editable": True, "description": "Max chat-history characters included."},
    "history_msg_limit": {"label": "History message limit", "group": "Chat", "min": 0, "max": 100, "advanced": False, "editable": True, "description": "Max prior messages included."},

    # Catalog
    "catalog_enabled":   {"label": "Catalog / list mode", "group": "Catalog", "advanced": False, "editable": True, "description": "Enable deterministic list/catalog mode."},
    "catalog_clarify_threshold": {"label": "Catalog clarify threshold", "group": "Catalog", "min": 1, "max": 1000, "advanced": False, "editable": True, "description": "Ambiguity threshold for list disambiguation."},
    "catalog_max_rows":  {"label": "Catalog max rows", "group": "Catalog", "min": 10, "max": 5000, "advanced": False, "editable": True, "description": "Max rows in a rendered catalog table."},
    "catalog_semantic_pool": {"label": "Catalog semantic pool", "group": "Catalog", "min": 10, "max": 5000, "advanced": False, "editable": True, "description": "BM25/dense pool size for catalog ranking."},

    # Summary
    "summary_strategy":  {"label": "Summary strategy", "group": "Summary", "advanced": False, "editable": True, "description": "map_reduce (deep) or single pass."},
    "summary_max_chunks": {"label": "Summary max chunks", "group": "Summary", "min": 5, "max": 500, "advanced": False, "editable": True, "description": "Max chunks fed into a deep summary."},
    "summary_context_budget": {"label": "Summary context budget", "group": "Summary", "min": 500, "max": 50000, "advanced": False, "editable": True, "description": "Context budget for summaries."},

    # Quiz
    "quiz_default_count": {"label": "Quiz default count", "group": "Quiz", "min": 1, "max": 100, "advanced": False, "editable": True, "description": "Default number of questions per quiz."},
    "quiz_output_format": {"label": "Quiz output format", "group": "Quiz", "advanced": True, "editable": True, "description": "Default export format when /api/quiz/<id>/export is called with no 'format' param: markdown | gift | csv | json | apkg."},
    "quiz_material_chunks": {"label": "Quiz material chunks", "group": "Quiz", "min": 1, "max": 100, "advanced": True, "editable": True, "description": "Max chunks of quiz material per prompt."},
    "quiz_material_words": {"label": "Quiz material words", "group": "Quiz", "min": 10, "max": 5000, "advanced": True, "editable": True, "description": "Max words of quiz material per prompt."},
    "quiz_depth":       {"label": "Quiz depth", "group": "Quiz", "advanced": True, "editable": True, "description": "Planner depth knob: surface | balanced | deep."},
    "quiz_topic_max_works": {"label": "Quiz topic max works", "group": "Quiz", "min": 1, "max": 20, "advanced": True, "editable": True, "description": "Max works a topic plan ranks units across."},
    "quiz_topic_section_pool": {"label": "Quiz topic section pool", "group": "Quiz", "min": 10, "max": 5000, "advanced": True, "editable": True, "description": "Section pool capped when cosine-ranking a topic plan."},
    # Quiz generation + audit pipeline (SESSION 3)
    "quiz_batch_parents": {"label": "Quiz batch parents", "group": "Quiz", "min": 1, "max": 50, "advanced": True, "editable": True, "description": "Parents sent per generation/verification batch."},
    "quiz_parent_word_cap": {"label": "Quiz parent word cap", "group": "Quiz", "min": 100, "max": 5000, "advanced": True, "editable": True, "description": "Per-parent word cap when building generation material."},
    "quiz_max_batches": {"label": "Quiz max batches", "group": "Quiz", "min": 1, "max": 200, "advanced": True, "editable": True, "description": "Max parent-batches per unit before giving up on the allocation."},
    "quiz_sample_children": {"label": "Sample children count", "group": "Quiz", "min": 5, "max": 500, "advanced": True, "editable": True, "description": "For non-parent_child books: number of evenly-spaced chunks to sample from Chroma (paginates directly). Higher = broader coverage but larger prompts."},
    "quiz_verify_pass": {"label": "Quiz verify pass", "group": "Quiz", "advanced": True, "editable": True, "description": "Run the inline LLM answer-key verification stage (true/false)."},
    "quiz_grounding_ratio": {"label": "Quiz grounding ratio", "group": "Quiz", "min": 0, "max": 1, "advanced": True, "editable": True, "description": "Min difflib ratio for an excerpt to be considered grounded in its parent."},
    "quiz_dedupe_jaccard": {"label": "Quiz dedupe Jaccard", "group": "Quiz", "min": 0, "max": 1, "advanced": True, "editable": True, "description": "Stemmed-token Jaccard at/above which a question is dropped as a duplicate."},
    "llm_structured_output": {"label": "LLM structured output", "group": "LLM", "advanced": True, "editable": True, "description": "Capability probe mode: auto | on | off for response_format json_schema."},
    "quiz_grade_llm": {"label": "Quiz LLM grading", "group": "Quiz", "advanced": True, "editable": True, "description": "Use the LLM to grade short-answer questions (true/false)."},
    "quiz_weak_threshold": {"label": "Quiz weak threshold", "group": "Quiz", "min": 0, "max": 1, "advanced": True, "editable": True, "description": "Unit accuracy below this is flagged as a weak area in analytics."},
    "fsrs_enabled": {"label": "FSRS review loop", "group": "Quiz", "advanced": True, "editable": True, "description": "Enroll missed questions into the shared FSRS spaced-repetition review store (true/false)."},
    "quiz_review_limit": {"label": "Review queue limit", "group": "Quiz", "min": 1, "max": 500, "advanced": True, "editable": True, "description": "Max due review cards returned by the review queue in one request."},

    # Retrieval
    "retrieval_top_k":   {"label": "Retrieval top-k", "group": "Retrieval", "min": 1, "max": 50, "advanced": False, "editable": True, "description": "Chunks surfaced in the prompt (capped 8)."},
    "relevance_threshold": {"label": "Relevance threshold", "group": "Retrieval", "min": 0, "max": 1, "advanced": False, "editable": True, "description": "Low-relevance guard score floor."},
    "max_retrieval_hops": {"label": "Retrieval hops", "group": "Retrieval", "min": 1, "max": 6, "advanced": False, "editable": True, "description": "Extra LLM-driven search rounds."},
    "retrieval_hops_driver": {"label": "Retrieval hops driver", "group": "Retrieval", "advanced": False, "editable": True, "description": "How multi-hop retrieval is driven (llm)."},

    # Rerank / hybrid (advanced, not in Setup UI)
    "rerank_model":      {"label": "Reranker model", "group": "Advanced", "advanced": True, "editable": True, "description": "Cross-encoder model id used for re-scoring."},
    "rerank_enabled":    {"label": "Rerank enabled", "group": "Advanced", "advanced": True, "editable": True, "description": "Enable cross-encoder reranking."},
    "fuse_rrf_k":        {"label": "Fuse RRF k", "group": "Advanced", "min": 1, "max": 200, "advanced": True, "editable": True, "description": "RRF rank-fusion constant k (default 60)."},
    "bm25_top_n":        {"label": "BM25 top n", "group": "Advanced", "min": 1, "max": 500, "advanced": True, "editable": True, "description": "BM25 hits to take as the lexical leg (default 30)."},
    "bm25_batch":        {"label": "BM25 batch", "group": "Advanced", "min": 100, "max": 100000, "advanced": True, "editable": True, "description": "Rows per Chroma pagination when building BM25 (default 20000)."},
    "bm25_min_coverage": {"label": "BM25 min coverage", "group": "Advanced", "min": 0, "max": 1, "advanced": True, "editable": True, "description": "Min hydrated coverage to trust persisted BM25 (default 0.9)."},
    "rerank_top_n":      {"label": "Rerank top n", "group": "Advanced", "min": 1, "max": 500, "advanced": True, "editable": True, "description": "How many fused candidates the reranker keeps (default 24)."},
    "max_chunks_per_title": {"label": "Max chunks per title", "group": "Advanced", "min": 1, "max": 50, "advanced": True, "editable": True, "description": "Cap on chunks kept per distinct title (default 3)."},
    "dense_pool_multiplier": {"label": "Dense pool multiplier", "group": "Advanced", "min": 1, "max": 10, "advanced": True, "editable": True, "description": "Dense retrieval pool = top_k * this (default 3)."},
    "fused_pool_floor":  {"label": "Fused pool floor", "group": "Advanced", "min": 1, "max": 500, "advanced": True, "editable": True, "description": "Lower bound on the fused RRF pool size (default 40)."},
    "lexical_backend":   {"label": "Lexical backend", "group": "Advanced", "advanced": True, "editable": True, "description": "BM25 backend: auto / bm25_sqlite / bm25_chroma."},

    # Fiction / OCR
    "fiction_tags":      {"label": "Fiction tags", "group": "OCR", "advanced": False, "editable": True, "description": "Calibre tags treated as fiction (comma-separated)."},
    "ocr_enabled":       {"label": "OCR enabled", "group": "OCR", "advanced": False, "editable": True, "description": "OCR scanned PDFs during ingest."},
    "ocr_merge":         {"label": "OCR write-back", "group": "OCR", "advanced": False, "editable": True, "description": "Merge OCR text into the original PDF (true) or sidecar (false)."},
    "ocr_backend":       {"label": "OCR backend", "group": "OCR", "advanced": False, "editable": True, "description": "OCR engine: tesseract or rapidocr."},
    "ocr_char_threshold": {"label": "OCR char threshold", "group": "OCR", "min": 0, "max": 10000, "advanced": True, "editable": False, "description": "Min characters to consider a page OCR-able (managed)."},
    "ocr_languages":     {"label": "OCR languages", "group": "OCR", "advanced": False, "editable": True, "description": "Comma-separated OCR language codes."},

    # Managed
    "sets":              {"label": "Sets", "group": "Managed", "advanced": True, "editable": False, "description": "Per-set ingest config (managed)."},
}


def _config_writable_keys() -> set:
    """The set of config keys the POST /api/config allowlist accepts.

    Derived from CONFIG_META so new keys can never drift out of sync: every
    registered key that is editable (not a managed value) is writable.
    """
    return {k for k, m in CONFIG_META.items() if m.get("editable", True)}


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


def _llm_native_states(cfg) -> dict | None:
    """Read per-model load state from LM Studio's native API.

    LM Studio JIT-loads models on the first request, so probing a model (even
    with ``max_tokens=5``) can trigger a full load. The native
    ``/api/v0/models`` endpoint reports state without touching any model. It is
    fail-soft: returns ``None`` when the endpoint is absent (remote/cloud
    providers or older servers), so callers can distinguish "unknown" from
    "known not-loaded".
    """
    import urllib.request
    base = str(cfg.get("llm_base_url", "")).rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + "/api/v0/models"
    headers = {}
    api_key = cfg.get("llm_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=2) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    states = {}
    for m in data.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        states[mid] = m.get("state") or ("loaded" if m.get("loaded") else "not-loaded")
    return states


def check_llm():
    """Check if the configured LLM API (LM Studio / llama.cpp / cloud) is
    reachable and list its models, plus (best-effort) which are loaded."""
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
        result = {"online": True, "models": models, "url": cfg["llm_base_url"]}
    except Exception as e:
        return {"online": False, "models": [], "error": str(e),
                "url": cfg["llm_base_url"],
                "model_states": {}, "loaded": None, "native": False}
    states = _llm_native_states(cfg)
    result["model_states"] = states or {}
    result["native"] = states is not None
    if states is None:
        result["loaded"] = None
    else:
        result["loaded"] = states.get(cfg.get("llm_model", "default")) == "loaded"
    return result


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


@app.route("/api/config/schema", methods=["GET"])
def api_config_schema():
    """Return the full {key: meta} registry for tools/agents to introspect."""
    return jsonify(CONFIG_META)


@app.route("/api/config", methods=["POST"])
def api_config_set():
    data = request.get_json(force=True)
    cfg = load_cfg()
    # Only allow whitelisted keys to update; the allowlist is derived from the
    # CONFIG_META registry so new keys can never drift out of sync.
    allowed = _config_writable_keys()
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


# Explicit probe cache (Setup "Test model" button). NEVER triggered on a status
# poll: a max_tokens=5 completion can JIT-load an unloaded model (slow, VRAM
# churn), so probing is always user-initiated and cached for 30s.
_LLM_PROBE_CACHE = {"at": 0.0, "model": None, "result": None}
_LLM_PROBE_TTL = 30


@app.route("/api/llm/test", methods=["POST"])
def api_llm_test():
    cfg = load_cfg()
    model = cfg.get("llm_model", "default")
    now = time.time()
    c = _LLM_PROBE_CACHE
    if c["result"] is not None and c["model"] == model \
            and (now - c["at"]) < _LLM_PROBE_TTL:
        return jsonify(c["result"])
    result = {"ok": False, "loaded": False, "model": model, "error": None}
    try:
        client = get_chat_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        text = (resp.choices[0].message.content or "").strip()
        result.update(ok=True, loaded=True, sample=text[:80])
    except Exception as e:
        result["error"] = str(e)
    _LLM_PROBE_CACHE.update(at=now, model=model, result=result)
    return jsonify(result)


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


# ─── Catalog / list mode ─────────────────────────────────────────────────────
#
# Deterministic, LLM-free path for "list books about X" style requests. It
# enumerates manifest.db titles and renders a Markdown table in Python, so the
# answer is complete and reliable instead of being capped by passage retrieval.

CATALOG_FIRST_PAGE = 20
_PENDING_CATALOG = {}          # conv_id -> pending dict (fast-path fallback)
_PENDING_LOCK = threading.Lock()

_ALL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:show\s+|list\s+|give\s+me\s+|see\s+)?"
    r"(?:all(?:\s+of\s+them)?|everything|the\s+(?:full|complete|entire)\s+"
    r"(?:list|thing)|full\s+list|complete\s+list|yes,?\s+please|yes|yep|sure|ok(?:ay)?)"
    r"\s*[.!]?\s*$", re.IGNORECASE)

_ALL_WORDS_RE = re.compile(
    r"\b(all|every|full|complete|entire|everything)\b", re.IGNORECASE)


def _last_assistant_meta(conv_id):
    conv = chat_store.get_conversation(conv_id)
    if not conv:
        return None
    for m in reversed(conv.get("messages", [])):
        if m.get("role") == "assistant":
            return m.get("meta") or {}
    return None


def _pending_followup(conv_id, query):
    """Resolve an "all"/"show all" follow-up against a persisted clarification."""
    if not conv_id or not _ALL_FOLLOWUP_RE.match(query or ""):
        return None
    pending = chat_store.get_pending_catalog(conv_id)
    if not pending:
        with _PENDING_LOCK:
            pending = _PENDING_CATALOG.get(conv_id)
    if not pending:
        return None
    meta = _last_assistant_meta(conv_id) or {}
    if not meta.get("clarify"):
        return None
    return pending


def _clear_pending_catalog(conv_id):
    if not conv_id:
        return
    chat_store.set_pending_catalog(conv_id, None)
    with _PENDING_LOCK:
        _PENDING_CATALOG.pop(conv_id, None)


def _set_pending_catalog(conv_id, pending):
    if not conv_id:
        return
    chat_store.set_pending_catalog(conv_id, pending)
    with _PENDING_LOCK:
        _PENDING_CATALOG[conv_id] = pending


def _catalog_prose(books, topic, cfg):
    """Optional short prose summary of the catalog result (best-effort)."""
    titles = ["- " + b.get("title", "") for b in books[:30]]
    ctx = "\n".join(titles)
    prompt = (f"The user asked about books in their library"
              + (f" on the topic '{topic}'" if topic else "")
              + ". In 2-4 sentences, describe what these books collectively "
                "cover and note any notable clusters. Do not invent titles "
                "beyond this list.\n\nTitles:\n" + ctx)
    try:
        client = get_chat_client()
        resp = client.chat.completions.create(
            model=cfg.get("llm_model", "default"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=int(cfg.get("max_tokens_summary", 4096) or 4096),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[catalog] prose summary failed: {e}", flush=True)
        return ""


def maybe_catalog(set_name, query, conv_id=None, filter_kind=None):
    """Handle a catalog/list request deterministically.

    Returns a result dict (with ``list_mode`` meta) or ``None`` when the query
    is not a catalog request (caller should fall through to RAG).
    """
    cfg = load_cfg()
    if not cfg.get("catalog_enabled", True):
        return None

    pending = _pending_followup(conv_id, query)
    if pending is not None:
        topic = pending.get("topic", "")
        fk = pending.get("filter_kind")
        explicit_all = True
        from_pending = True
    else:
        if not CAT.is_list_intent(query):
            return None
        topic = CAT.extract_topic(query)
        fk = CAT.extract_filter_kind(query) or filter_kind
        explicit_all = bool(_ALL_WORDS_RE.search(query or ""))
        from_pending = False

    try:
        collection = get_chat_collection(set_name)
    except SystemExit:
        return None
    if collection.count() == 0:
        return None

    try:
        books = CAT.find_books(topic, set_name, fk, cfg, collection=collection,
                               embedder=get_chat_embedder())
    except Exception as e:
        print(f"[catalog] find_books failed: {e}", flush=True)
        return None

    label = "Library catalog"
    if fk:
        label += f" ({fk})"
    if topic:
        label += f" — books about **{topic}**"

    if not books:
        _clear_pending_catalog(conv_id)
        msg = (f"### {label}\n\nNo books on **{topic}** were found in this "
               "collection." if topic else
               f"### {label}\n\nNo books were found in this collection.")
        msg += ("\n\nTry a broader topic, a different keyword, or confirm the "
                "set has been indexed (Ingest tab).")
        return {"answer": msg, "sources": [], "list_mode": True, "count": 0,
                "shown": 0, "truncated": False, "clarify": False,
                "fiction_only": False, "low_relevance": False,
                "relevance_reason": ""}

    threshold = int(cfg.get("catalog_clarify_threshold", CATALOG_FIRST_PAGE)
                    or CATALOG_FIRST_PAGE)
    max_rows = int(cfg.get("catalog_max_rows", 200) or 200)
    total = len(books)
    clarify = False
    if explicit_all or total <= threshold:
        shown_books = books[:max_rows]
        truncated = total > max_rows
    else:
        shown_books = books[:min(CATALOG_FIRST_PAGE, total)]
        truncated = True
        clarify = True

    answer = f"### {label}\n\nFound **{total}** matching title(s).\n\n"
    answer += CAT.render_table(shown_books)
    if clarify:
        answer += (f"\n\n**{total - len(shown_books)} more** — reply **all** "
                   "for the full list.")
    elif truncated:
        answer += (f"\n\n_Showing the first {len(shown_books)} of {total} "
                   f"(catalog_max_rows={max_rows})._")

    if CAT.has_summary_keywords(query) and shown_books:
        prose = _catalog_prose(shown_books, topic, cfg)
        if prose:
            answer += "\n\n" + prose

    if clarify:
        _set_pending_catalog(conv_id, {"topic": topic, "count": total,
                                       "filter_kind": fk})
    else:
        _clear_pending_catalog(conv_id)

    return {
        "answer": answer, "sources": [], "list_mode": True, "count": total,
        "shown": len(shown_books), "truncated": truncated or clarify,
        "clarify": clarify, "fiction_only": False, "low_relevance": False,
        "relevance_reason": "",
    }


def _turn_meta(result):
    """Extract the extra meta keys persisted alongside an assistant turn."""
    meta = {
        "sources": result.get("sources", []),
        "fiction_only": result.get("fiction_only", False),
        "low_relevance": result.get("low_relevance", False),
        "relevance_reason": result.get("relevance_reason", ""),
    }
    for k in ("list_mode", "count", "shown", "truncated", "clarify",
              "quiz_mode", "quiz_count", "quiz_spec_proposal", "quiz_spec_id",
              "quiz_spec_confirmed"):
        if k in result:
            meta[k] = result[k]
    return meta


# ─── Quiz planning (SESSION 2) ───────────────────────────────────────────────
#
# The SESSION 2 planning layer: a deterministic section inventory + one cached
# LLM outline pass (quiz_syllabus) and a planner (quiz_plan) that resolves
# scope (work / sections / topic), applies a depth knob, allocates question
# counts proportional to section length, and records out-of-scope refusals to
# the ledger. No question generation happens yet (that is SESSION 3); these
# endpoints only plan and persist an editable QuizSpec.

def _set_name_from_request():
    return (request.get_json(force=True, silent=True) or {}).get("set_name") or "veracrypt1"


def _quiz_plan_collection(set_name):
    """Return (collection, embedder) for planner topic ranking; None-guarded."""
    try:
        collection = get_chat_collection(set_name)
    except SystemExit:
        collection = None
    embedder = None
    if collection is not None and collection.count() > 0:
        embedder = get_chat_embedder()
    return collection, embedder


@app.route("/api/quiz/plan", methods=["POST"])
def api_quiz_plan():
    """Resolve a quiz request (work / sections / topic) into an editable spec."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or _set_name_from_request()
    cfg = load_cfg()
    collection, embedder = _quiz_plan_collection(set_name)
    try:
        result = quiz_plan.plan_quiz(
            data, set_name, cfg, embedder=embedder, collection=collection)
    except Exception as e:
        print(f"[quiz/plan] failed: {e}", flush=True)
        return jsonify({"error": f"Quiz planning failed: {e}"}), 500
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)


@app.route("/api/quiz/plan/revise", methods=["POST"])
def api_quiz_plan_revise():
    """Apply conversational edits to a stored quiz spec (include/exclude/count…)."""
    data = request.get_json(force=True, silent=True) or {}
    spec_id = data.get("spec_id") or (data.get("spec") or {}).get("id")
    spec = data.get("spec")
    if spec is None and spec_id:
        spec = quiz_store.get_spec(spec_id)
    if not spec:
        return jsonify({"error": "No spec provided."}), 400
    deltas = data.get("deltas") or data
    try:
        result = quiz_plan.revise_spec(spec, deltas)
    except Exception as e:
        print(f"[quiz/plan/revise] failed: {e}", flush=True)
        return jsonify({"error": f"Spec revision failed: {e}"}), 500
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result)


@app.route("/api/quiz/syllabus", methods=["POST"])
def api_quiz_syllabus():
    """Build/return a work's cached syllabus (for the Study-tab builder)."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "veracrypt1"
    title = (data.get("work") or data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "No work title provided."}), 400
    cfg = load_cfg()
    try:
        syllabus = quiz_syllabus.get_syllabus(
            title, set_name, cfg, get_chat_client())
    except Exception as e:
        print(f"[quiz/syllabus] failed: {e}", flush=True)
        return jsonify({"error": f"Syllabus failed: {e}"}), 500
    return jsonify(syllabus)


@app.route("/api/quiz/specs", methods=["GET"])
def api_quiz_specs():
    set_name = request.args.get("set_name") or "veracrypt1"
    return jsonify({"specs": quiz_store.list_specs(set_name)})


@app.route("/api/quiz/spec/<spec_id>", methods=["GET"])
def api_quiz_spec_get(spec_id):
    spec = quiz_store.get_spec(spec_id)
    if not spec:
        return jsonify({"error": "Spec not found."}), 404
    return jsonify(spec)


@app.route("/api/quiz/spec/<spec_id>", methods=["DELETE"])
def api_quiz_spec_delete(spec_id):
    quiz_store.delete_spec(spec_id)
    return jsonify({"ok": True})


@app.route("/api/quiz/out-of-scope", methods=["GET"])
def api_quiz_out_of_scope():
    return jsonify({"entries": quiz_store.read_out_of_scope()})


@app.route("/api/quiz/books", methods=["GET"])
def api_quiz_books():
    """List the works in a set (titles) for the Study-tab work picker."""
    set_name = request.args.get("set_name") or "veracrypt1"
    collection = None
    try:
        collection = get_chat_collection(set_name)
    except SystemExit:
        pass
    if collection is None or collection.count() == 0:
        return jsonify({"books": []})
    books = CAT.list_books(set_name, collection)
    return jsonify({"books": [{"title": b["title"], "kind": b.get("kind", "")}
                               for b in books if b.get("title")]})


# ─── Quiz generation pipeline (SESSION 3) ───────────────────────────────────
#
# Work-specific generation: material comes from an ordered walk of the
# `parents` table (never semantic search). Each batch is generated and put
# through the 4-stage audit (scripts/quiz_audit.py) — parse sanity, grounding
# against the real parent text, inline LLM answer-key verification, and cross-
# batch dedupe — with drop + top-up. The build runs in a worker thread and
# streams SSE ``progress`` (heartbeat per unit) and ``delta`` ({type: unit}/
# {type: done}) events via a queue, mirroring the chat stream pattern.
# Generation is blocked while an ingest is running (same reason as /api/chat:
# avoid ChromaDB concurrent read/write errors).

@app.route("/api/quiz/generate", methods=["POST"])
def api_quiz_generate():
    """Generate a quiz from a persisted QuizSpec. SSE progress events."""
    if ingest_active():
        return jsonify({"error": "An ingest is running; wait for it to finish "
                                 "before generating a quiz."}), 409
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "veracrypt1"
    spec_id = data.get("spec_id")
    spec = data.get("spec")
    if spec is None and spec_id:
        spec = quiz_store.get_spec(spec_id)
    if not spec:
        return jsonify({"error": "No spec provided."}), 400
    cfg = load_cfg()
    return Response(
        _quiz_generate_sse(set_name, spec, cfg),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no"},
    )


def _quiz_generate_sse(set_name, spec, cfg):
    import json as _json
    import queue as _queue
    evq = _queue.Queue()

    def _worker():
        try:
            result = quiz_build.build_quiz(
                spec, set_name, cfg, get_chat_client(),
                stream_cb=lambda ev: evq.put(("delta", ev)),
                progress_cb=lambda msg: evq.put(("progress", {"message": msg})),
                collection=get_chat_collection(set_name),
                embedder=get_chat_embedder(),
            )
            evq.put(("done", result))
        except SystemExit:
            evq.put(("error", "Collection not available for this set."))
        except Exception as e:
            print(f"[quiz/generate] failed: {e}", flush=True)
            evq.put(("error", str(e)))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while True:
        kind, payload = evq.get()
        if kind == "progress":
            yield "event: progress\ndata: " + _json.dumps(payload) + "\n\n"
        elif kind == "delta":
            # Follow the chat deltas convention: a top-level object.
            yield "event: delta\ndata: " + _json.dumps({"wave": payload}) + "\n\n"
        elif kind == "error":
            yield "event: error\ndata: " + _json.dumps({"error": payload}) + "\n\n"
            return
        elif kind == "done":
            yield "event: done\ndata: " + _json.dumps(payload) + "\n\n"
            return


@app.route("/api/quizzes", methods=["GET"])
def api_quizzes_list():
    """List generated quizzes (status, counts) for the Study tab."""
    set_name = request.args.get("set_name") or "veracrypt1"
    return jsonify({"quizzes": quiz_store.list_quizzes(set_name)})


@app.route("/api/quiz/<quiz_id>", methods=["GET"])
def api_quiz_get(quiz_id):
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    return jsonify(quiz)


@app.route("/api/quiz/<quiz_id>/export", methods=["GET"])
def api_quiz_export(quiz_id):
    """Export a stored quiz in a portable format (markdown|gift|csv|json|apkg).

    Wires the `quiz_output_format` config key as the default when no
    ``format`` query param is given (SESSION 6)."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    fmt = request.args.get("format") or str(
        load_cfg().get("quiz_output_format", "markdown") or "markdown")
    try:
        payload, mime, is_binary = quiz_export.export_quiz(quiz, fmt)
    except ImportError as e:
        return jsonify({"error": str(e)}), 400
    fname = f"quiz-{quiz.get('id', quiz_id)}.{'apkg' if fmt == 'apkg' else 'gift' if fmt == 'gift' else 'csv' if fmt == 'csv' else 'json' if fmt == 'json' else 'md'}"
    return Response(payload, mimetype=mime,
                    headers={"Content-Disposition":
                             f'attachment; filename="{fname}"'})


@app.route("/api/quiz/<quiz_id>", methods=["DELETE"])
def api_quiz_delete(quiz_id):
    quiz_store.delete_quiz(quiz_id)
    return jsonify({"ok": True})


# ─── Quiz player, grading & analytics (SESSION 4) ────────────────────────────
#
# The Study-tab player takes a stored quiz one question at a time. Grading is
# deterministic for mcq / true_false / fill_blank and LLM-assisted for
# short_answer (scripts/quiz_grade.py), using only each question's own
# provenance excerpt. Attempts (answers, scores, timestamps) are persisted
# INSIDE the quiz file under quiz["attempts"] so an interrupted session is not
# lost and the result survives a server restart. Analytics aggregate per-unit /
# per-difficulty / per-type accuracy and provenance-linked study points.

def _quiz_attempt_id():
    import uuid as _uuid
    return _uuid.uuid4().hex[:12]


def _strip_answers(question):
    """Return a question for the player WITHOUT revealing the answer."""
    q = dict(question)
    q.pop("answer", None)
    return q


@app.route("/api/quiz/<quiz_id>/attempt", methods=["POST"])
def api_quiz_attempt(quiz_id):
    """Start a quiz attempt: register it and return the questions sans answers."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    questions = [q for q in (quiz.get("questions") or []) if q.get("q")]
    if not questions:
        return jsonify({"error": "This quiz has no questions."}), 400
    attempt_id = _quiz_attempt_id()
    attempts = quiz.setdefault("attempts", [])
    attempts.append({
        "attempt_id": attempt_id,
        "started": int(time.time()),
        "grades": {},
        "finished": False,
        "analytics": None,
    })
    quiz_store.update_quiz(quiz)
    return jsonify({
        "attempt_id": attempt_id,
        "total": len(questions),
        "questions": [_strip_answers(q) for q in questions],
    })


@app.route("/api/quiz/<quiz_id>/answer", methods=["POST"])
def api_quiz_answer(quiz_id):
    """Grade a single answer and store it in the current attempt."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    data = request.get_json(force=True, silent=True) or {}
    attempt_id = data.get("attempt_id")
    qid = str(data.get("qid"))
    response = data.get("response")
    if not attempt_id or qid is None:
        return jsonify({"error": "attempt_id and qid are required."}), 400
    attempts = quiz.get("attempts") or []
    attempt = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
    if not attempt:
        return jsonify({"error": "Attempt not found."}), 404
    question = next((q for q in (quiz.get("questions") or [])
                     if str(q.get("qid")) == qid), None)
    if not question:
        return jsonify({"error": "Question not found."}), 404
    cfg = load_cfg()
    graded = quiz_grade.grade_answer(question, response, cfg, get_chat_client())
    if graded is None:
        return jsonify({"error": "Could not grade this answer."}), 400
    attempt.setdefault("grades", {})[qid] = {
        "response": response,
        "correct": graded["correct"],
        "expected": graded["expected"],
        "feedback": graded["feedback"],
        "score": graded["score"],
        "ts": int(time.time()),
    }
    quiz_store.update_quiz(quiz)
    return jsonify({
        "correct": graded["correct"],
        "expected": graded["expected"],
        "feedback": graded["feedback"],
        "score": graded["score"],
    })


@app.route("/api/quiz/<quiz_id>/finish", methods=["POST"])
def api_quiz_finish(quiz_id):
    """Finish an attempt: compute analytics and persist the attempt."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    data = request.get_json(force=True, silent=True) or {}
    attempt_id = data.get("attempt_id")
    if not attempt_id:
        return jsonify({"error": "attempt_id is required."}), 400
    attempts = quiz.get("attempts") or []
    attempt = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
    if not attempt:
        return jsonify({"error": "Attempt not found."}), 404
    analytics = quiz_grade.compute_analytics(
        attempt.get("grades") or {}, quiz.get("questions") or [],
        weak_threshold=cfg_float(load_cfg(), "quiz_weak_threshold", 0.6, 0, 1))
    # Enroll missed questions into the FSRS review store (SESSION 5) so they
    # resurface on a spaced-repetition schedule in the Review pane. Gated by
    # the fsrs_enabled config key; idempotent per (quiz_id, qid).
    enrolled = 0
    if load_cfg().get("fsrs_enabled", True):
        # Mirror quiz_grade.enumerate_all's qid rule (qid else array index) so
        # study points line back up even for legacy questions without a qid.
        by_qid = {str(q.get("qid") or i): q
                  for i, q in enumerate(quiz.get("questions") or [])}
        for sp in analytics.get("study_points") or []:
            qid = str(sp.get("qid"))
            q = by_qid.get(qid)
            if q and quiz_store.enroll_review_question(
                    quiz_id, quiz.get("set_name") or "", qid, q,
                    already_correct=False):
                enrolled += 1
    attempt["finished"] = True
    attempt["analytics"] = analytics
    attempt["finished_at"] = int(time.time())
    quiz_store.update_quiz(quiz)
    return jsonify({"analytics": analytics, "enrolled_to_review": enrolled})


@app.route("/api/quiz/<quiz_id>/analytics", methods=["GET"])
def api_quiz_analytics(quiz_id):
    """Return the analytics for a finished attempt (latest by default)."""
    quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found."}), 404
    attempt_id = request.args.get("attempt_id")
    attempts = [a for a in (quiz.get("attempts") or [])
                if a.get("finished") and a.get("analytics")]
    if not attempts:
        return jsonify({"analytics": None})
    if attempt_id:
        hit = next((a for a in attempts if a.get("attempt_id") == attempt_id), None)
        if hit:
            return jsonify({"analytics": hit["analytics"]})
        return jsonify({"analytics": None})
    last = sorted(attempts, key=lambda a: a.get("finished_at", 0)).pop()
    return jsonify({"analytics": last["analytics"]})


# ─── FSRS review loop (SESSION 5) ────────────────────────────────────────────
#
# Missed questions are enrolled into a single shared review store
# (quizzes/review.json, keyed by quiz_id:qid) at attempt finish so the
# due-today queue spans quizzes. The Review pane pulls cards from
# GET /api/review/queue, applies an FSRS rating via
# POST /api/review/<quiz_id>/<qid>, and shows counts + strength-over-time via
# GET /api/review/stats. Scheduling is computed by py-fsrs with UTC now.

@app.route("/api/review/queue", methods=["GET"])
def api_review_queue():
    """Due review cards across quizzes (question + provenance), oldest-due
    first, for one set."""
    set_name = request.args.get("set") or request.args.get("set_name") or "veracrypt1"
    limit = cfg_int(load_cfg(), "quiz_review_limit", 50, 1, 500)
    cards = quiz_store.fetch_due_reviews(set_name=set_name, limit=limit)
    out = []
    for c in cards:
        question = c.get("question") or {}
        prov = question.get("provenance") or {}
        out.append({
            "key": c.get("key"),
            "quiz_id": c.get("quiz_id"),
            "qid": c.get("qid"),
            "set_name": c.get("set_name"),
            "type": c.get("type") or question.get("type") or "mcq",
            "question": question.get("q") or "",
            "choices": question.get("choices") or [],
            "answer": question.get("answer") or "",
            "section_title": prov.get("section_title") or "",
            "excerpt": (prov.get("excerpt_snippet") or "")[:400],
            "difficulty": question.get("difficulty") or "",
            "due": (c.get("fsrs") or {}).get("due"),
            "last_rating": c.get("last_rating"),
            "reviews": len(c.get("history") or []),
        })
    return jsonify({"cards": out, "count": len(out)})


@app.route("/api/review/<quiz_id>/<qid>", methods=["POST"])
def api_review_rating(quiz_id, qid):
    """Apply an FSRS rating (again|hard|good|easy) to a stored review card."""
    if not load_cfg().get("fsrs_enabled", True):
        return jsonify({"error": "FSRS review is disabled in config."}), 400
    data = request.get_json(force=True, silent=True) or {}
    rating = data.get("rating") or request.form.get("rating")
    card = quiz_store.record_review_rating(quiz_id, qid, rating)
    if card is None:
        if rating not in ("again", "hard", "good", "easy"):
            return jsonify({"error": "Rating must be again|hard|good|easy."}), 400
        return jsonify({"error": "Review card not found (or FSRS unavailable)."}), 404
    history = card.get("history") or []
    last = history[-1] if history else {}
    return jsonify({
        "ok": True,
        "rating": rating,
        "due": (card.get("fsrs") or {}).get("due"),
        "stability": last.get("stability"),
        "last_rating": card.get("last_rating"),
    })


@app.route("/api/review/stats", methods=["GET"])
def api_review_stats():
    """Review counts, retention rate, and a strength-over-time series."""
    set_name = request.args.get("set") or request.args.get("set_name") or "veracrypt1"
    return jsonify(quiz_store.fetch_review_stats(set_name))


@app.route("/api/review/summary", methods=["GET"])
def api_review_summary():
    """Per-quiz due counts (for the due badge on the generated-quiz list)."""
    set_name = request.args.get("set") or request.args.get("set_name") or "veracrypt1"
    cards = quiz_store.fetch_due_reviews(set_name=set_name, limit=100000)
    counts = {}
    for c in cards:
        qid = c.get("quiz_id")
        counts[qid] = counts.get(qid, 0) + 1
    return jsonify({"due": counts})


# ─── Course view (SESSION 6) ─────────────────────────────────────────────────
#
# Surfaces the SESSION 2 syllabus artifact as a browsable course: a work's unit
# list with reading summaries (whole-work via catalog.map_reduce_summary, or a
# per-unit LLM summary over that unit's parents) and a per-unit "quiz this unit"
# action that plans a sections-scope spec. Simple unit progress is derived from
# stored quizzes (how many of the work's units have generated questions).

def _course_quizzed_counts(set_name, work):
    """Return {normalized unit_title: count} of generated-quiz questions that
    cite this work, so the course can mark units as "quizzed"."""
    import re as _re
    counts = {}
    norm = lambda s: _re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())
    for q in quiz_store.list_quizzes(set_name):
        for qu in (q.get("questions") or []):
            prov = qu.get("provenance") or {}
            if prov.get("title") and norm(work) and \
               (norm(prov["title"]) in norm(work) or norm(work) in norm(prov["title"])):
                key = qu.get("section_title") or prov.get("section_title") or ""
                if key:
                    counts[norm(key)] = counts.get(norm(key), 0) + 1
    return counts


@app.route("/api/course", methods=["GET"])
def api_course():
    """Course/syllabus browser for one work: its unit list (with word counts)
    and a ``quizzed`` flag per unit derived from stored quizzes."""
    set_name = request.args.get("set") or request.args.get("set_name") or "veracrypt1"
    work = (request.args.get("work") or "").strip()
    if not work:
        return jsonify({"error": "a work title is required"}), 400
    syllabus = quiz_syllabus.get_syllabus(work, set_name)
    units = syllabus.get("units") or []
    counts = _course_quizzed_counts(set_name, work)
    norm = lambda s: __import__("re").sub(
        r"[^a-z0-9]+", "", (s or "").strip().lower())
    out_units = []
    for u in units:
        quizzed = bool(counts.get(norm(u.get("title"))))
        out_units.append({
            "unit_id": u.get("unit_id"),
            "title": u.get("title"),
            "ordinal": u.get("ordinal"),
            "parent_ids": u.get("parent_ids") or [],
            "subtopics": u.get("subtopics") or [],
            "words": u.get("words") or 0,
            "quizzed": quizzed,
        })
    return jsonify({
        "title": work, "set_name": set_name,
        "units": out_units,
        "unit_count": len(out_units),
        "quizzed_count": sum(1 for u in out_units if u["quizzed"]),
        "warnings": syllabus.get("warnings") or [],
    })


@app.route("/api/course/summary", methods=["POST"])
def api_course_summary():
    """Whole-work deep reading summary via catalog.map_reduce_summary."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "veracrypt1"
    work = (data.get("work") or "").strip()
    if not work:
        return jsonify({"error": "a work title is required"}), 400
    try:
        collection = get_chat_collection(set_name)
        if collection.count() == 0:
            return jsonify({"error": "no collection available"}), 400
        import agent as _a
        matched = _a._match_titles(work, set_name, limit=4)
        if not matched:
            return jsonify({"error": f"no work matching {work!r}"}), 404
        summary = map_reduce_summary(
            set_name, matched, load_cfg(), collection, get_chat_embedder(),
            get_chat_client())
        return jsonify({"work": work, "set_name": set_name,
                        "matched": matched, "summary": summary})
    except SystemExit:
        return jsonify({"error": "collection unavailable"}), 400
    except Exception as e:
        print(f"[course] summary failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/course/unit-summary", methods=["POST"])
def api_course_unit_summary():
    """Short per-unit reading summary over that unit's parents via an LLM call."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "veracrypt1"
    work = (data.get("work") or "").strip()
    unit_id = (data.get("unit_id") or "").strip()
    if not work or not unit_id:
        return jsonify({"error": "work and unit_id are required"}), 400
    syllabus = quiz_syllabus.get_syllabus(work, set_name)
    unit = next((u for u in (syllabus.get("units") or [])
                 if u.get("unit_id") == unit_id), None)
    if not unit:
        return jsonify({"error": "unit not found"}), 404
    parent_ids = unit.get("parent_ids") or []
    import agent as _a
    texts = _a._load_parent_texts(parent_ids, set_name) if parent_ids else {}
    parts = []
    for pid in parent_ids:
        p = texts.get(pid)
        if p:
            parts.append(" ".join((p.get("text") or "").split())[:900])
    if not parts:
        return jsonify({"error": "no parent text for this unit"}), 404
    cfg = load_cfg()
    prompt = (
        f"Write a concise reading summary (2-4 sentences) of the chapter/section "
        f"\"{unit.get('title')}\" of \"{work}\", covering its main points.\n\n"
        f"--- Section text ---\n\n" + "\n\n".join(parts))
    try:
        resp = get_chat_client().chat.completions.create(
            model=cfg.get("llm_model", "default"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=int(cfg.get("max_tokens_summary", 4096) or 4096),
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[course] unit summary failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    return jsonify({"work": work, "unit_id": unit_id,
                    "title": unit.get("title"), "summary": text})


@app.route("/api/course/quiz-unit", methods=["POST"])
def api_course_quiz_unit():
    """Plan (and persist) a sections-scope spec for a single unit, so the UI can
    jump straight to generating a quiz on that unit."""
    data = request.get_json(force=True, silent=True) or {}
    set_name = data.get("set_name") or "veracrypt1"
    work = (data.get("work") or "").strip()
    unit_id = (data.get("unit_id") or "").strip()
    count = int(data.get("count") or 10)
    if not work or not unit_id:
        return jsonify({"error": "work and unit_id are required"}), 400
    try:
        collection = get_chat_collection(set_name)
        plan = quiz_plan.plan_quiz(
            {"mode": "sections", "work": work, "unit_ids": [unit_id],
             "count": count}, set_name, load_cfg(),
            embedder=get_chat_embedder(), collection=collection)
    except SystemExit:
        return jsonify({"error": "collection unavailable"}), 400
    except Exception as e:
        print(f"[course] quiz-unit plan failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    if "error" in plan:
        return jsonify({"error": plan["error"]}), 400
    spec = plan["spec"]
    if not spec.get("id"):
        spec = quiz_store.create_spec(spec)
    return jsonify({"spec": spec})


# ─── Quiz / study mode ───────────────────────────────────────────────────────
#
# "Quiz me on X" / "make flashcards about Y" routes to structured question
# generation (scripts/study.py) instead of the normal QA path. Quiz intent
# takes priority over catalog/list intent. Material is retrieved by title
# match when a work is named, or via the topic-based catalog search otherwise.

# Keep the quiz-material context within an 8k-token model's budget.
# System prompt + formatting ≈ 300 tokens, generation quota ≈ 512 tokens →
# material budget ≈ 7,400 tokens. With MC=10 × MW=500 each chunk ~250 tokens,
# total ≈ 5,800 tokens which fits comfortably with margin for output.
QUIZ_MATERIAL_CHUNKS = 10
QUIZ_MATERIAL_WORDS = 500


def _persist_pending_quiz_spec(conv_id, spec):
    """Persist a quiz-spec proposal for conversational revision (SESSION 6)."""
    if not conv_id:
        return
    chat_store.set_pending_quiz_spec(conv_id, spec)


def _clear_pending_quiz_spec(conv_id):
    if conv_id:
        chat_store.set_pending_quiz_spec(conv_id, None)


def _spec_proposal_body(spec, set_name, action="planned"):
    """Render a spec proposal with numbered options for the chat reply."""
    units = spec.get("units") or []
    title = spec.get("title") or ""
    head = f"### 🧠 Quiz plan"
    if title:
        head += f" — {title}"
    head += f"  (set: {set_name})\n\n"
    body = (head + f"Here's a proposal for a **{spec.get('count')}-question** "
            f"quiz ({spec.get('mode')} · {spec.get('depth')}).\n\n")
    for i, u in enumerate(units, 1):
        body += f"**{i}. {u.get('title')}** — {u.get('allocation')} question(s)\n"
    for w in (spec.get("warnings") or [])[:4]:
        body += f"\n> ⚠ {w}"
    body += ("\n\nReply to adjust, e.g. **\"drop unit 1\"**, **\"double unit 2\"**, "
             "**\"make it 30 questions\"**, **\"harder\"**, **\"deep\"**, or "
             "**\"build it\"** to generate.")
    return body


def _quiz_spec_proposal(set_name, topic, count, cfg, conv_id, filter_kind=None):
    """Build a QuizSpec proposal for a quiz request and stash it as the
    conversation's pending_quiz_spec. Returns a result dict (``quiz_mode`` +
    ``quiz_spec_proposal``) or None if no collection/plan is available."""
    try:
        collection = get_chat_collection(set_name)
        if collection.count() == 0:
            return None
        plan = quiz_plan.plan_quiz(
            {"mode": "topic", "topic": topic, "count": count,
             "difficulty": "mixed"}, set_name, cfg,
            embedder=get_chat_embedder(), collection=collection)
    except SystemExit:
        return None
    except Exception as e:
        print(f"[quiz] spec proposal failed: {e}", flush=True)
        return None
    if "error" in plan or plan.get("out_of_scope"):
        return {
            "answer": (f"### 🧠 Quiz\n\nCould not plan a quiz on **{topic}**: "
                       f"{plan.get('error') or plan.get('refusal')}"),
            "sources": [], "quiz_mode": True, "quiz_count": 0,
            "fiction_only": False, "low_relevance": False,
            "relevance_reason": "",
        }
    spec = plan["spec"]
    if not spec.get("id"):
        spec = quiz_store.create_spec(spec)
    else:
        persisted = quiz_store.get_spec(spec.get("id"))
        if persisted:
            spec = persisted
    _persist_pending_quiz_spec(conv_id, spec)
    return {
        "answer": _spec_proposal_body(spec, set_name),
        "sources": [], "quiz_mode": True, "quiz_count": spec.get("count", count),
        "quiz_spec_proposal": True, "quiz_spec_id": spec.get("id"),
        "fiction_only": False, "low_relevance": False, "relevance_reason": "",
    }


def _resolve_pending_spec_delta(conv_id, query):
    """If this conversation has a pending_quiz_spec proposal and the query
    reads as an adjustment to it, apply quiz_plan.revise_spec and return a
    result dict. Returns None when there is no applicable pending spec."""
    if not conv_id:
        return None
    pending = chat_store.get_pending_quiz_spec(conv_id)
    if not pending:
        return None
    q = (query or "").strip().lower()
    if not q:
        return None
    # A plain "build it / generate / go" confirms the pending spec.
    if re.search(r"\b(build|generate|go ahead|make it now|yes|yep|confirm)\b", q):
        return {"confirm": True, "spec": pending,
                "answer": ("Proceeding to build the pending quiz. "
                           "Open the Study tab's Generated quizzes to run it."),
                "sources": [], "quiz_mode": True, "quiz_count": 0,
                "quiz_spec_confirmed": True, "fiction_only": False,
                "low_relevance": False, "relevance_reason": ""}
    # Detect explicit adjustment keywords; without them fall through.
    if not re.search(r"\b(unit|units|drop|exclude|remove|add|include|double|"
                     r"half|more|fewer|questions?|harder|easier|deep|surface|"
                     r"balanced|difficulty|count|make it|\d+)\b", q):
        return None
    deltas = {}
    # count: "make it N questions" / "N questions"
    m = re.search(r"\b(\d+)\s*(questions?|q)\b", q)
    if m and not re.search(r"\bunit", q):
        deltas["count"] = int(m.group(1))
    # difficulty
    if re.search(r"\bharder\b", q):
        deltas["difficulty"] = "hard"
    elif re.search(r"\beasier\b", q):
        deltas["difficulty"] = "easy"
    # depth
    for depth in ("deep", "surface", "balanced"):
        if re.search(rf"\b{depth}\b", q):
            deltas["depth"] = depth
            break
    if not deltas:
        return None
    result = quiz_plan.revise_spec(pending, deltas)
    spec = (result or {}).get("spec") or pending
    _persist_pending_quiz_spec(conv_id, spec)
    return {
        "answer": _spec_proposal_body(spec, spec.get("set_name") or "veracrypt1",
                                      action="revised"),
        "sources": [], "quiz_mode": True,
        "quiz_count": spec.get("count", 0), "quiz_spec_proposal": True,
        "quiz_spec_id": spec.get("id"), "fiction_only": False,
        "low_relevance": False, "relevance_reason": "",
    }


def maybe_quiz(set_name, query, conv_id=None, filter_kind=None):
    """Handle a quiz/test request. Returns a result dict (with ``quiz_mode``)
    or ``None`` when the query is not a quiz request."""
    cfg = load_cfg()
    mc = cfg_int(cfg, "quiz_material_chunks", QUIZ_MATERIAL_CHUNKS, 1, 100)
    mw = cfg_int(cfg, "quiz_material_words", QUIZ_MATERIAL_WORDS, 10, 5000)
    if not CAT.is_quiz_intent(query):
        return None

    topic = CAT.extract_topic(query) or ""
    count = CAT.extract_count(query, int(cfg.get("quiz_default_count", 10) or 10))
    fk = CAT.extract_filter_kind(query) or filter_kind

    # SESSION 6 conversational negotiation: for a larger quiz request (or one
    # that names a work and wants a reviewable quiz) we route through the
    # planner to produce an editable spec proposal with numbered options,
    # stored as pending_quiz_spec so follow-up messages can revise it. Small
    # requests keep the fast inline quiz (no regression).
    propose = count > int(cfg.get("quiz_default_count", 10) or 10) or \
        bool(re.search(r"\b(plan|build|save|long|big|review(able)?)\b",
                       query or "", re.IGNORECASE))
    if propose:
        proposal = _quiz_spec_proposal(set_name, topic, count, cfg, conv_id,
                                       fk)
        if proposal is not None:
            return proposal

    try:
        collection = get_chat_collection(set_name)
    except SystemExit:
        return None
    if collection.count() == 0:
        return None

    import agent as _agent_mod
    embedder = get_chat_embedder()
    material = None
    matched = _agent_mod._match_titles(query or topic, set_name)
    if matched:
        where = {"title": {"$in": matched}}
        try:
            hits = _agent_mod.retrieve(", ".join(matched), embedder, collection,
                                       top_k=mc, cfg=cfg,
                                       where_extra=where)
        except Exception:
            hits = []
        material = _agent_mod.build_context(hits[:mc],
                                            max_words=mw)
    else:
        books = CAT.find_books(topic, set_name, fk, cfg,
                               collection=collection, embedder=embedder)
        titles = [b["title"] for b in books[:5]]
        if titles:
            where = {"title": {"$in": titles}}
            try:
                hits = _agent_mod.retrieve(topic, embedder, collection,
                                           top_k=mc, cfg=cfg,
                                           where_extra=where)
            except Exception:
                hits = []
            material = _agent_mod.build_context(hits[:mc],
                                                max_words=mw)

    if not material:
        return {"answer": (f"### 🧠 Quiz\n\nNo material was found to quiz you "
                           f"on{f' regarding **{topic}**' if topic else ''} in "
                           "this collection."),
                "sources": [], "quiz_mode": True, "quiz_count": 0,
                "fiction_only": False, "low_relevance": False,
                "relevance_reason": ""}

    label = topic or ", ".join(matched) or "your library"
    questions, err = STUDY.generate_quiz(label, material, cfg, get_chat_client())
    if not questions:
        body = (f"### 🧠 Quiz\n\nA quiz on **{label}** could not be generated."
                + (f" ({err})" if err else ""))
    else:
        body = f"### 🧠 Quiz — {label}\n\n" + STUDY.render_markdown(questions)

    return {"answer": body, "sources": [], "quiz_mode": True,
            "quiz_count": len(questions),
            "fiction_only": False, "low_relevance": False,
            "relevance_reason": ""}


# ─── Map-reduce deep summaries ───────────────────────────────────────────────
#
# When summary_strategy == "map_reduce" and the user names one or more works
# (title_mode), we produce a deeper book-wide summary instead of stuffing a
# capped excerpt pool into one prompt. The implementation lives in catalog.py
# (map child chunks -> section parents spread across the book, batch map calls,
# single reduce call). The reduce is one LLM call, so it streams token-by-token.
# These thin wrappers keep the catalog/summary concerns out of the rag path.

def map_reduce_summary(set_name, matched_titles, cfg, collection, embedder,
                       client, progress_cb=None, stream_cb=None):
    """Run the two-pass map-reduce summary (see catalog.py)."""
    return CAT.map_reduce_summary(
        set_name, matched_titles, cfg, collection, embedder, client,
        progress_cb=progress_cb, stream_cb=stream_cb)


def _single_shot_summary(set_name, matched_titles, cfg, collection, embedder,
                         client):
    return CAT._single_shot_summary(
        set_name, matched_titles, cfg, collection, embedder, client)


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

    # Deep map-reduce summaries for named works (non-streaming callers).
    if title_mode and (cfg.get("summary_strategy") or "single") == "map_reduce":
        try:
            answer = map_reduce_summary(
                set_name, matched_titles, cfg, collection, embedder,
                get_chat_client())
            if not answer:
                answer = "Unable to produce a summary for the named work(s)."
            return {"answer": answer, "sources": sources,
                    "fiction_only": fiction_only,
                    "low_relevance": low_rel, "relevance_reason": low_reason,
                    "deep_summary": True}
        except Exception as e:
            print(f"[chat] map-reduce summary failed ({e}); "
                  "falling back to single-pass", flush=True)

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
# Long tables/quizzes need headroom so follow-ups like "explain answer 3" work.
HISTORY_MSG_LIMIT = 12
HISTORY_CHAR_BUDGET = 10000
HISTORY_MSG_CHAR_CAP = 2500


def _conversation_history(conv_id) -> list[dict]:
    """Load prior user/assistant messages (oldest first) within size limits."""
    if not conv_id:
        return []
    conv = chat_store.get_conversation(conv_id)
    if not conv:
        return []
    cfg = load_cfg()
    msg_limit = int(cfg.get("history_msg_limit", HISTORY_MSG_LIMIT) or HISTORY_MSG_LIMIT)
    char_budget = int(cfg.get("history_char_budget", HISTORY_CHAR_BUDGET) or HISTORY_CHAR_BUDGET)
    msgs = [m for m in conv.get("messages", [])
            if m.get("role") in ("user", "assistant") and m.get("content")]
    msgs = msgs[-msg_limit:]
    history, used = [], 0
    for m in msgs:
        content = (m.get("content") or "")[:HISTORY_MSG_CHAR_CAP]
        used += len(content) + 40
        if used > char_budget:
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


def _persist_chat(conv_id, query, result):
    """Persist a user/assistant turn to the conversation store on success."""
    if not conv_id or "error" in result:
        return
    chat_store.add_message(conv_id, "user", query)
    chat_store.add_message(conv_id, "assistant", result.get("answer", ""),
                           _turn_meta(result))


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming RAG chat for the web UI.

    SSE events: ``delta`` (raw text), ``done`` (final JSON incl. sources/meta),
    ``progress`` (optional heartbeat), ``error`` (JSON with an ``error``
    string). Aborting mid-stream skips persistence. Under ``chat_mode:
    "agentic"`` this runs the streaming agentic loop (progress heartbeats +
    streamed final answer); otherwise it stays single-shot.
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

    quiz = maybe_quiz(set_name, query, conv_id, filter_kind)
    if quiz is not None:
        text = quiz.get("answer", "")
        yield "event: delta\ndata: " + _json.dumps(text) + "\n\n"
        yield "event: done\ndata: " + _json.dumps(quiz) + "\n\n"
        _persist_chat(conv_id, query, quiz)
        return

    delta = _resolve_pending_spec_delta(conv_id, query)
    if delta is not None:
        text = delta.get("answer", "")
        yield "event: delta\ndata: " + _json.dumps(text) + "\n\n"
        yield "event: done\ndata: " + _json.dumps(delta) + "\n\n"
        _persist_chat(conv_id, query, delta)
        return

    cat = maybe_catalog(set_name, query, conv_id, filter_kind)
    if cat is not None:
        text = cat.get("answer", "")
        yield "event: delta\ndata: " + _json.dumps(text) + "\n\n"
        yield "event: done\ndata: " + _json.dumps(cat) + "\n\n"
        _persist_chat(conv_id, query, cat)
        return

    # Streaming map-reduce deep summaries: show progress during map batches,
    # then stream the single reduce call token-by-token. The map/reduce work
    # runs in a worker thread that pushes progress/delta events into a queue;
    # this generator drains the queue and yields them as live SSE events.
    if (cfg.get("summary_strategy") or "single") == "map_reduce":
        collection = None
        _matched = []
        try:
            collection = get_chat_collection(set_name)
            import agent as _agent_mod
            _matched = _agent_mod._match_titles(query, set_name)
        except SystemExit:
            pass
        if _matched and collection is not None:
            import queue as _queue
            evq = _queue.Queue()

            def _worker():
                try:
                    def _progress_cb(msg):
                        evq.put(("progress", {"message": msg}))
                    def _stream_cb(t):
                        evq.put(("delta", t))
                    map_reduce_summary(
                        set_name, _matched, cfg, collection,
                        get_chat_embedder(), get_chat_client(),
                        progress_cb=_progress_cb, stream_cb=_stream_cb)
                    evq.put(("done", True))
                except Exception as e:
                    print(f"[chat] streamed map-reduce failed ({e})", flush=True)
                    evq.put(("error", str(e)))

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            parts = []
            while True:
                kind, data = evq.get()
                if kind == "progress":
                    yield "event: progress\ndata: " + _json.dumps(
                        {"message": data["message"]}) + "\n\n"
                elif kind == "delta":
                    parts.append(data)
                    yield "event: delta\ndata: " + _json.dumps(data) + "\n\n"
                elif kind == "error":
                    yield "event: error\ndata: " + _json.dumps(
                        {"error": data}) + "\n\n"
                    return
                elif kind == "done":
                    break
            text = "".join(parts).strip()
            if not text:
                yield "event: error\ndata: " + _json.dumps(
                    {"error": "LLM returned no usable summary."}) + "\n\n"
                return
            done = {"answer": text, "sources": [],
                    "fiction_only": False, "low_relevance": False,
                    "relevance_reason": "", "deep_summary": True}
            yield "event: done\ndata: " + _json.dumps(done) + "\n\n"
            _persist_chat(conv_id, query, done)
            return

    # Streaming agentic loop: when enabled and chat_mode != "single", run the
    # tool loop in a worker thread. Progress heartbeats fire during tool
    # resolution; the final answer is regenerated as one streamed completion
    # whose deltas flow back as SSE. Falls back to the single-shot path below.
    if cfg.get("agentic_enabled", True) and \
            (cfg.get("chat_mode") or "agentic") != "single":
        import queue as _queue
        evq = _queue.Queue()

        def _agent_worker():
            try:
                agent_loop_mod = __import__("agent_loop",
                                            fromlist=["run_agent_loop"])
                result = agent_loop_mod.run_agent_loop(
                    set_name, query, history, cfg, get_chat_client(),
                    top_k=top_k, filter_kind=filter_kind,
                    embedder=get_chat_embedder(),
                    collection=get_chat_collection(set_name),
                    reranker=get_chat_reranker(),
                    progress_cb=lambda msg: evq.put(("progress", {"message": msg})),
                    stream_cb=lambda t: evq.put(("delta", t)),
                )
                evq.put(("done", result))
            except Exception as e:
                print(f"[chat] streamed agentic loop failed ({e}); "
                      "using single-shot", flush=True)
                evq.put(("error", str(e)))

        t = threading.Thread(target=_agent_worker, daemon=True)
        t.start()
        agent_parts = []
        agent_done = None
        while True:
            kind, data = evq.get()
            if kind == "progress":
                yield "event: progress\ndata: " + _json.dumps(
                    {"message": data["message"]}) + "\n\n"
            elif kind == "delta":
                agent_parts.append(data)
                yield "event: delta\ndata: " + _json.dumps(data) + "\n\n"
            elif kind == "error":
                yield "event: error\ndata: " + _json.dumps(
                    {"error": data}) + "\n\n"
                return
            elif kind == "done":
                agent_done = data
                break
        answer = "".join(agent_parts).strip()
        if not answer and agent_done and agent_done.get("answer"):
            answer = agent_done["answer"].strip()
        if agent_done and agent_done.get("error"):
            yield "event: error\ndata: " + _json.dumps(
                {"error": agent_done["error"]}) + "\n\n"
            return
        if not answer:
            yield "event: error\ndata: " + _json.dumps(
                {"error": "LLM returned no usable answer. Is the LLM API "
                          "running?"}) + "\n\n"
            return
        done = {"answer": answer, "sources": agent_done.get("sources", []),
                "fiction_only": agent_done.get("fiction_only", False),
                "low_relevance": agent_done.get("low_relevance", False),
                "relevance_reason": agent_done.get("relevance_reason", "")}
        yield "event: done\ndata: " + _json.dumps(done) + "\n\n"
        _persist_chat(conv_id, query, done)
        return

    payload = _prepare_rag(set_name, query, top_k, filter_kind, history)
    if "error" in payload:
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
    quiz = maybe_quiz(set_name, query, None, filter_kind)
    if quiz is not None:
        for word in quiz["answer"].split(" "):
            yield "data: " + _json.dumps({
                "choices": [{"delta": {"content": word + " "}}]}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    cat = maybe_catalog(set_name, query, None, filter_kind)
    if cat is not None:
        for word in cat["answer"].split(" "):
            yield "data: " + _json.dumps({
                "choices": [{"delta": {"content": word + " "}}]}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
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


def _has_table(conn, name):
    import sqlite3
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _rename_manifest_set(old: str, new: str):
    """Point every manifest row for `old` at `new` so a later re-ingest
    doesn't double-count already-indexed files."""
    db = RAG_ROOT / "manifest.db"
    if not db.exists():
        return
    import sqlite3
    conn = sqlite3.connect(str(db))
    # Re-pointing old rows onto `new` can collide (rel_path, set_name) PK with
    # rows `new` already owns; prefer the old set's rows and drop the dupes.
    conn.execute(
        "DELETE FROM files WHERE set_name = ? AND rel_path IN "
        "(SELECT rel_path FROM files WHERE set_name = ?)", (new, old))
    conn.execute("UPDATE files SET set_name = ? WHERE set_name = ?", (new, old))
    if _has_table(conn, "parents"):
        conn.execute(
            "DELETE FROM parents WHERE set_name = ? AND parent_id IN "
            "(SELECT parent_id FROM parents WHERE set_name = ?)", (new, old))
        conn.execute("UPDATE parents SET set_name = ? WHERE set_name = ?", (new, old))
    for tbl in ("bm25_tokens", "bm25_df"):
        if _has_table(conn, tbl):
            conn.execute(
                f"UPDATE {tbl} SET set_name = ? WHERE set_name = ?", (new, old))
    if _has_table(conn, "meta"):
        conn.execute("UPDATE meta SET key = ? WHERE key = ?", (f"target:{new}", f"target:{old}"))
    conn.commit()
    conn.close()


def _drop_manifest_set(set_name: str):
    db = RAG_ROOT / "manifest.db"
    if not db.exists():
        return
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM files WHERE set_name = ?", (set_name,))
    if _has_table(conn, "parents"):
        conn.execute("DELETE FROM parents WHERE set_name = ?", (set_name,))
    for tbl in ("bm25_tokens", "bm25_df"):
        if _has_table(conn, tbl):
            conn.execute(f"DELETE FROM {tbl} WHERE set_name = ?", (set_name,))
    if _has_table(conn, "meta"):
        conn.execute("DELETE FROM meta WHERE key = ?", (f"target:{set_name}",))
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
