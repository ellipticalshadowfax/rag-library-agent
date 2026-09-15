#!/usr/bin/env python3
"""mcp_server.py - Expose the RAG library index to MCP-capable chat clients
(LM Studio, etc.) as callable tools.

Rather than the RAG engine producing the whole answer, the local model in the
chat client (e.g. LM Studio) drives the conversation and calls these tools on
demand to pull relevant excerpts from the vector index, then grounds its reply
in them. Embeddings still run on CPU; the embedder and Chroma collection are
loaded lazily once per process and cached.

Tools:
  - search_library(query, set_name, top_k, filter_kind)
        Vector-search the index and return the top matching excerpts.
  - summarize_work(title, set_name, top_k)
        Route the query straight to one named work's chunks for a summary.
  - list_collections()
        List available index collections (set names) and chunk counts.

Transports:
  stdio (default)  ->  LM Studio "Local" MCP server
  http             ->  LM Studio "Remote" MCP server (--http --port 8765 /mcp)

Run:  .venv/bin/python scripts/mcp_server.py            # stdio
      .venv/bin/python scripts/mcp_server.py --http      # streamable-http on :8765/mcp
"""

import argparse
import contextlib
import os
import re
import sqlite3
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for embeddings

from mcp.server.fastmcp import FastMCP

from _paths import rag_root

RAG_ROOT = rag_root()
DEFAULT_SET = "veracrypt1"
_QUERY_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "at", "is", "are", "was", "were", "what", "which", "how",
    "book", "books", "summarize", "summary", "about", "explain",
}


@contextlib.contextmanager
def _muted_stdout():
    """Keep the stdio JSON-RPC channel clean: any stray print() (e.g. the
    embedder's "Loading…" banner) is diverted to stderr so it can't corrupt
    the protocol stream."""
    real = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real


def _load_agent():
    import agent
    return agent


# ── Cached shared resources (embedder + chroma) ─────────────────────────────

_resources = {}


def _embedder():
    if "embedder" not in _resources:
        agent = _load_agent()
        with _muted_stdout():
            _resources["embedder"] = agent.setup_embedder(agent.load_config())
    return _resources["embedder"]


def _collection(set_name: str):
    key = f"col:{set_name}"
    if key not in _resources:
        import chromadb
        with _muted_stdout():
            client = chromadb.PersistentClient(path=str(RAG_ROOT / "index"))
            names = [c.name for c in client.list_collections()]
        if set_name not in names:
            avail = ", ".join(names) if names else "(none — run ingest first)"
            raise ValueError(f"collection '{set_name}' not found. Available: {avail}")
        _resources[key] = client.get_collection(set_name)
    return _resources[key]


def default_set_name() -> str:
    cfg = _load_agent().load_config()
    if cfg.get("sets"):
        return next(iter(cfg["sets"]))
    return DEFAULT_SET


def _reranker():
    """Lazy-load the cross-encoder reranker so MCP search matches the web
    server's retrieval quality (hybrid fusion + rerank). Returns None when
    disabled or unavailable."""
    key = "reranker"
    if key not in _resources:
        cfg = _load_agent().load_config()
        if not cfg.get("rerank_enabled", True):
            _resources[key] = None
        else:
            try:
                from sentence_transformers import CrossEncoder
                model = cfg.get("rerank_model")
                with _muted_stdout():
                    _resources[key] = CrossEncoder(
                        model, device=cfg.get("embed_device", "cpu"))
            except Exception as e:
                print(f"[mcp] reranker unavailable: {e}")
                _resources[key] = None
    return _resources[key]


def list_collection_names() -> list[str]:
    import chromadb
    client = chromadb.PersistentClient(path=str(RAG_ROOT / "index"))
    return [c.name for c in client.list_collections()]


def _match_titles(query: str, set_name: str, limit: int = 4) -> list[str]:
    """Match the query against known titles in this set's manifest.db, so a
    request that names a work routes straight to that book's chunks."""
    db = RAG_ROOT / "manifest.db"
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT DISTINCT title FROM files WHERE set_name=? "
            "AND title IS NOT NULL AND title<>''", (set_name,)).fetchall()
        con.close()
    except Exception as e:
        print(f"[mcp] manifest title lookup failed: {e}")
        return []

    q = re.sub(r"[^a-z0-9 ]", " ", query.lower()).strip()
    qwords = {w for w in q.split() if w not in _QUERY_STOP}
    scored = []
    for (title,) in rows:
        t = re.sub(r"[^a-z0-9 ]", " ", title.lower()).strip()
        if len(t) < 3:
            continue
        twords = {w for w in t.split() if w not in _QUERY_STOP}
        if t in q or q in t:
            score = 100 + len(t)
        elif len(twords) >= 2 and len(twords & qwords) >= 2:
            score = 60 + len(t)
        elif len(twords & qwords) == 1 and len((twords & qwords).pop()) >= 5:
            score = 40 + len(t)
        else:
            continue
        scored.append((score, title))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, t in scored[:limit]]


def _render_hits(hits: list) -> str:
    """Format retrieved chunks as markdown grounding text for the model."""
    lines = []
    for i, h in enumerate(hits):
        meta = h["metadata"] or {}
        kind = meta.get("kind", "unknown")
        kind_label = "[FICTION]" if kind == "fiction" else "[NON-FICTION]"
        title = meta.get("title", "Unknown")
        source = f"{title} ({meta.get('source', 'unknown')})"
        page = meta.get("page", "")
        page_str = f", p.{page}" if page else ""
        score = h.get("rerank_score")
        if score is None and h.get("distance") is not None:
            score = round(1 - h["distance"], 3)
        elif score is not None:
            score = round(float(score), 3)
        score_str = f" (score {score:.3f})" if score is not None else ""
        # Surface the chapter/section heading when parent-child retrieval
        # collapsed a child to its parent section — lets the model cite the
        # exact part of the book (and stay on-topic) instead of the whole work.
        section = meta.get("section_title") or ""
        section_str = f" — {section.strip()}" if section and section.strip() else ""
        lines.append(f"## Source {i+1} {kind_label}{score_str} — {source}{section_str}{page_str}")
        doc = h.get("document", "")
        lines.append(doc)
        lines.append("")
    return "\n".join(lines)


# ── Tools ────────────────────────────────────────────────────────────────────

mcp = FastMCP("rag-library", instructions=(
    "You are chatting with the user against their personal e-book library. "
    "Use search_library to retrieve relevant excerpts before answering, and "
    "cite the source title from the returned excerpts. Excerpts tagged "
    "[FICTION] are fiction and must never be presented as fact. If retrieval "
    "turns up no strong match, say so and answer from your own knowledge.\n\n"
    "STOP RULE: Perform AT MOST ONE search_library call per user question, then "
    "answer immediately from what it returned. Do NOT call search_library "
    "again with the same or a reworded query — the library is index-only, "
    "re-searching the same topic returns the same excerpts. Re-reading the "
    "results never reveals new books; instead answer with what you have and, "
    "if coverage is genuinely missing, tell the user plainly. Only issue a "
    "second search if the user explicitly names a DIFFERENT book or topic. "
    "Never loop, never ask 'shall I search again'. A final answer is required."))


@mcp.tool()
def list_collections() -> str:
    """List the available library index collections (set names) and how many
    chunks each holds. Call this first if you need to know which set_name
    to pass to search_library or summarize_work."""
    out = []
    for name in list_collection_names():
        try:
            count = _collection(name).count()
            out.append(f"{name}: {count} chunks")
        except Exception:
            out.append(f"{name}: (error)")
    return "\n".join(out) if out else "(no collections found — run ingest first)"


@mcp.tool()
def search_library(query: str, set_name: str = "", top_k: int = 6,
                   filter_kind: str = "") -> str:
    """ALWAYS call this tool before answering any question about the user's books
    or library. It retrieves relevant excerpts from the user's e-book collection.
    Never answer from your own knowledge alone when the user asks about their
    library — search first, then cite the source titles from the results.

    Call ONCE per question and then answer — do not keep calling it. The library
    is index-only; re-searching the same or reworded query returns the same
    excerpts, so looping on this tool wastes turns and never finds new material.

    Args:
      query: the question or search phrase (natural language).
      set_name: which index collection to search (see list_collections). Leave empty for the default.
      top_k: how many excerpts to return (1-8).
      filter_kind: 'fiction' or 'nonfiction' to restrict results, or leave empty for all.
    """
    agent = _load_agent()
    cfg = agent.load_config()
    set_name = set_name or default_set_name()
    top_k = max(1, min(int(top_k), 8))
    collection = _collection(set_name)

    # Use the SAME full retrieval pipeline as the web server / eval harness:
    # title routing (query names a specific work -> route to its chunks),
    # hybrid dense+BM25 fusion, cross-encoder rerank, and parent-child
    # generation. This keeps MCP search quality on par with the web UI and
    # fixes gaps like "Which nations does Gulliver visit?" failing to surface
    # Gulliver's Travels.
    with _muted_stdout():
        rag = agent.retrieve_rag(
            set_name, query, top_k, filter_kind or None, cfg,
            embedder=_embedder(), collection=collection,
            client=None, reranker=_reranker(), backend=cfg.get("lexical_backend", "auto"))
    hits = rag["hits"]
    if not hits:
        return ("(No matching documents were retrieved from the library for "
                f"this query in set '{set_name}'.)")

    kept, used = [], 0
    budget = int(cfg.get("context_word_budget", 1000) or 1000)
    for h in hits:
        if used + len(h["document"].split()) > budget:
            break
        kept.append(h)
        used += len(h["document"].split())
    hits = kept

    fiction_only = rag["fiction_only"]
    low_rel = rag["low_relevance"]
    reason = rag["relevance_reason"]

    head = [f"# Library search: \"{query}\"  (set: {set_name})", ""]
    if fiction_only:
        head.append("NOTE: ALL retrieved sources are FICTION. Do NOT present "
                    "them as fact.")
        head.append("")
    if low_rel:
        note = reason or "low relevance"
        head.append("NOTE: Retrieval found no strong match "
                    f"({note}). Say the library lacks direct coverage and "
                    "answer from your own knowledge.")
        head.append("")
    out = "\n".join(head) + _render_hits(hits)
    out += ("\n\n[REMINDER] These are the library's best matches for this "
            "question. Answer now using them; re-searching the same topic "
            "will not surface different books.")
    return out


@mcp.tool()
def summarize_work(title: str, set_name: str = "", top_k: int = 8) -> str:
    """Retrieve excerpts of ONE named work (book) from the library so the model
    can summarize or discuss it specifically.

    Call ONCE for a given work and then answer; do not re-call for the same
    title — the returned excerpts are already the best of that book.

    Args:
      title: the title (or distinctive part of it) of the book.
      set_name: which index collection (see list_collections). Leave empty for the default.
      top_k: how many excerpts to return (1-10).
    """
    agent = _load_agent()
    cfg = agent.load_config()
    set_name = set_name or default_set_name()
    top_k = max(1, min(int(top_k), 10))
    collection = _collection(set_name)

    matched = _match_titles(title, set_name, limit=4)
    if not matched:
        return (f"(No work matching '{title}' was found in set '{set_name}'. "
                "Try a different title or use search_library instead.)")

    # Route through retrieve_rag's titled path so summaries get the same
    # hybrid + rerank + parent-child treatment as search / the web UI.
    with _muted_stdout():
        rag = agent.retrieve_rag(
            set_name, title, top_k, None, cfg,
            embedder=_embedder(), collection=collection,
            client=None, reranker=_reranker(),
            backend=cfg.get("lexical_backend", "auto"))
    hits = rag["hits"]

    fiction = [h for h in hits if h["metadata"].get("kind") == "fiction"]
    honorific = rag["matched_titles"] or matched
    head = [f"# Summarize requested work(s): {', '.join(honorific[:4])}  "
            f"(set: {set_name})", ""]
    if fiction:
        head.append("NOTE: These works are FICTION. Present them as fiction, "
                    "not fact.")
        head.append("")
    return "\n".join(head) + _render_hits(hits)


def main():
    parser = argparse.ArgumentParser(description="RAG library MCP server")
    parser.add_argument("--http", action="store_true",
                        help="Serve over SSE/HTTP instead of stdio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
