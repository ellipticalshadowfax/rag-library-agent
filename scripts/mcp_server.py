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

RAG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SET = "library"
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
        source = f"{meta.get('title', 'Unknown')} ({meta.get('source', 'unknown')})"
        page = meta.get("page", "")
        page_str = f", p.{page}" if page else ""
        score = 1 - h["distance"]
        lines.append(f"## Source {i+1} {kind_label} (score {score:.3f}) — {source}{page_str}")
        lines.append(h["document"])
        lines.append("")
    return "\n".join(lines)


# ── Tools ────────────────────────────────────────────────────────────────────

mcp = FastMCP("rag-library", instructions=(
    "You are chatting with the user against their personal e-book library. "
    "Use search_library to retrieve relevant excerpts before answering, and "
    "cite the source title from the returned excerpts. Excerpts tagged "
    "[FICTION] are fiction and must never be presented as fact. If retrieval "
    "turns up no strong match, say so and answer from your own knowledge."))


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

    pool_k = min(top_k * 3, 24)
    with _muted_stdout():
        hits = agent.retrieve(query, _embedder(), collection, top_k=pool_k,
                              filter_kind=filter_kind or None, cfg=cfg)
    hits = agent.diversify_hits(hits, limit=top_k)
    if not hits:
        return ("(No matching documents were retrieved from the library for "
                f"this query in set '{set_name}'.)")

    kept, used = [], 0
    for h in hits:
        if used + len(h["document"].split()) > 1500:
            break
        kept.append(h)
        used += len(h["document"].split())
    hits = kept

    fiction = [h for h in hits if h["metadata"].get("kind") == "fiction"]
    nonfiction = [h for h in hits if h["metadata"].get("kind") != "fiction"]
    low_rel, reason = agent.detect_low_relevance(query, hits, cfg)

    head = [f"# Library search: \"{query}\"  (set: {set_name})", ""]
    if fiction and not nonfiction:
        head.append("NOTE: ALL retrieved sources are FICTION. Do NOT present "
                    "them as fact.")
        head.append("")
    if low_rel:
        note = reason or "low relevance"
        head.append("NOTE: Retrieval found no strong match "
                    f"({note}). Say the library lacks direct coverage and "
                    "answer from your own knowledge.")
        head.append("")
    return "\n".join(head) + _render_hits(hits)


@mcp.tool()
def summarize_work(title: str, set_name: str = "", top_k: int = 8) -> str:
    """Retrieve excerpts of ONE named work (book) from the library so the model
    can summarize or discuss it specifically.

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

    where = {"title": {"$in": matched}}
    with _muted_stdout():
        hits = agent.retrieve(title, _embedder(), collection, top_k=min(top_k * 3, 24),
                              cfg=cfg, where_extra=where)
    hits = agent.diversify_hits(hits, limit=None, max_per_title=top_k)

    fiction = [h for h in hits if h["metadata"].get("kind") == "fiction"]
    head = [f"# Summarize requested work(s): {', '.join(matched)}  (set: {set_name})", ""]
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
