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

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Skip HuggingFace remote checks; models cached from first-run

from mcp.server.fastmcp import FastMCP

from _paths import rag_root, resolve_device

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
                        model, device=resolve_device(cfg.get("embed_device")))
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
    budget = int(cfg.get("context_word_budget", 3000) or 3000)
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
def list_books(topic: str = "", set_name: str = "", filter_kind: str = "") -> str:
    """Enumerate the library as a catalog table of titles (not passage
    excerpts). Use this for \"list/show/find/what books about ...\" requests
    where the user wants to browse titles rather than read any one work.

    Args:
      topic: topic phrase to filter by, or empty to list the entire library.
      set_name: which index collection (see list_collections). Leave empty for the default.
      filter_kind: 'fiction' or 'nonfiction' to restrict, or leave empty for all.
    """
    import catalog
    agent = _load_agent()
    cfg = agent.load_config()
    set_name = set_name or default_set_name()
    books = catalog.find_books(
        topic or "", set_name, filter_kind or None, cfg,
        collection=_collection(set_name), embedder=_embedder())
    if not books:
        return (f"(No books matching{f' topic {topic!r}' if topic else ''} "
                f"were found in set '{set_name}'.)")
    head = (f"# Library catalog{f' — books about **{topic}**' if topic else ''}"
            f"  (set: {set_name})\n\n"
            f"Found {len(books)} matching title(s).\n\n")
    out = head + catalog.render_table(books[:50])
    if len(books) > 50:
        out += f"\n\n_(truncated — {len(books) - 50} more not shown)_"
    return out


@mcp.tool()
def make_quiz(topic: str, set_name: str = "", count: int = 10) -> str:
    """Generate a structured practice quiz on a topic or a named work, with
    collapsible answers and per-question source citations.

    For a named book (title match), this paginates Chroma directly to gather
    ALL matching chunks — not just the top-k semantically similar ones — so
    the LLM gets full coverage of the book's content. For topic queries, it
    falls back to semantic search across multiple titles.

    Args:
      topic: the topic or book title to quiz on.
      set_name: which index collection (see list_collections). Leave empty for the default.
      count: how many questions to generate (1-50).
    """
    import catalog
    import study
    agent = _load_agent()
    cfg = agent.load_config()
    set_name = set_name or default_set_name()
    try:
        count = max(1, min(int(count or 10), 50))
    except (TypeError, ValueError):
        count = int(cfg.get("quiz_default_count", 10) or 10)
    collection = _collection(set_name)

    matched = agent._match_titles(topic, set_name, limit=4)
    material = None
    MC = max(1, min(int(cfg.get("quiz_material_chunks", 10) or 10), 100))
    MW = max(10, min(int(cfg.get("quiz_material_words", 500) or 500), 20000))
    with _muted_stdout():
        if matched:
            # Named book detected — paginate Chroma directly to get ALL chunks
            # for this book, bypassing semantic top-k limits.
            batch_size = 500
            all_hits = []
            offset = 0
            while True:
                where = {"title": {"$in": matched}}
                result = collection.get(offset=offset, limit=batch_size,
                                        include=["documents", "metadatas"],
                                        where=where)
                ids = result.get("ids") or []
                if not ids:
                    break
                docs = result.get("documents") or []
                metadatas = result.get("metadatas") or []
                for i, doc in enumerate(docs):
                    doc = (doc or "").strip()
                    meta = metadatas[i] or {}
                    # Skip bibliographic / metadata-only chunks
                    lower_doc = doc.lower()
                    if any(skip in lower_doc for skip in [
                            'isbn', 'bf575', 'to my wife', 'contents']):
                        continue
                    if len(doc.split()) < 30:
                        continue
                    all_hits.append({
                        "id": ids[i],
                        "document": doc,
                        "metadata": {
                            "title": meta.get("title", ""),
                            "source": meta.get("source", ""),
                            "section_title": meta.get("section_title", ""),
                        },
                    })
                offset += batch_size

            # Deduplicate by first-300-chars key
            seen = set()
            unique_hits = []
            for h in all_hits:
                key = h["document"][:300]
                if key not in seen:
                    seen.add(key)
                    unique_hits.append(h)

            # Shuffle for variety, then take up to MC chunks
            import random as _random
            _random.seed(42)
            _random.shuffle(unique_hits)
            hits = unique_hits[:MC]
            material = agent.build_context(hits[:MC], max_words=MW)
        else:
            books = catalog.find_books(topic, set_name, None, cfg,
                                       collection=collection, embedder=_embedder())
            titles = [b["title"] for b in books[:5]]
            if titles:
                where = {"title": {"$in": titles}}
                hits = agent.retrieve(topic, _embedder(), collection,
                                      top_k=MC, cfg=cfg, where_extra=where)
                material = agent.build_context(hits[:MC], max_words=MW)
    if not material:
        return (f"(No material found to quiz on{topic and f' topic {topic!r}' or ''}"
                f" in set '{set_name}'.)")
    saved = dict(cfg)
    saved["quiz_default_count"] = count
    questions, err = study.generate_quiz(topic or ", ".join(matched), material,
                                          saved, agent.setup_client(cfg))
    if not questions:
        return f"(Quiz generation failed: {err or 'no questions parsed'})"
    return (f"# Quiz{f' — {topic}' if topic else ''}  (set: {set_name})\n\n"
            + study.render_markdown(questions))


@mcp.tool()
def plan_quiz(request: dict, set_name: str = "") -> str:
    """Plan a persistent (stored, reviewable) quiz against a whole work,
    explicit sections, or a topic. Returns a spec proposal with numbered unit
    options. Use for 'quiz me on X' when a stored/reviewable quiz is wanted.

    Args:
      request: planning request, e.g. {"mode": "work|sections|topic",
        "work"/"topic": "...", "count": int}. Common keys: count, difficulty,
        types, depth, minimum_per_unit.
      set_name: which index collection. Leave empty for the default.
    """
    agent = _load_agent()
    cfg = agent.load_config()
    set_name = set_name or default_set_name()
    import quiz_plan
    with _muted_stdout():
        plan = quiz_plan.plan_quiz(
            request or {}, set_name, cfg,
            embedder=_embedder(), collection=_collection(set_name))
    if "error" in plan:
        return f"(plan_quiz failed: {plan['error']})"
    if plan.get("out_of_scope"):
        return f"(Out of scope: {plan.get('refusal') or 'no matching units'})"
    spec = plan["spec"]
    if not spec.get("id"):
        import quiz_store
        with _muted_stdout():
            spec = quiz_store.create_spec(spec)
    lines = [f"# Quiz plan: {spec.get('title')}  (set: {set_name})",
             f"- **Spec id**: `{spec.get('id')}`",
             f"- **Mode**: {spec.get('mode')} · **Depth**: {spec.get('depth')}",
             f"- **Total questions**: {spec.get('count')}", ""]
    for i, u in enumerate(spec.get("units") or [], 1):
        lines.append(f"{i}. {u.get('title')} — {u.get('allocation')} q"
                     f" (ordinal {u.get('ordinal')})")
    if (spec.get("warnings") or []):
        lines.append("")
        lines.append("Warnings: " + "; ".join(spec["warnings"][:4]))
    lines.append("")
    lines.append("Confirm with build_quiz(spec_id) or adjust first.")
    return "\n".join(lines)


@mcp.tool()
def build_quiz(spec_id: str = "", set_name: str = "") -> str:
    """Build (generate) a persistent quiz from a spec id returned by
    plan_quiz. Blocking; returns the quiz id, question count, and build report
    summary.

    Args:
      spec_id: the spec id from plan_quiz.
      set_name: which index collection. Leave empty for the default.
    """
    agent = _load_agent()
    cfg = agent.load_config()
    set_name = set_name or default_set_name()
    import quiz_store
    import quiz_build
    with _muted_stdout():
        spec = quiz_store.get_spec(spec_id)
        if not spec:
            return (f"(No spec found for id {spec_id!r}. Create one with "
                    "plan_quiz first.)")
        quiz = quiz_build.build_quiz(
            spec, set_name, cfg, agent.setup_client(cfg),
            collection=_collection(set_name), embedder=_embedder())
    report = quiz.get("report") or {}
    body = (f"# Quiz built: {quiz.get('title')}  (set: {set_name})\n\n"
            f"- **Quiz id**: `{quiz.get('id')}`\n"
            f"- **Questions**: {len(quiz.get('questions') or [])}\n"
            f"- **Status**: {quiz.get('status')}\n")
    if report:
        body += (f"- **Structured output**: {report.get('structured_output')}\n"
                 f"- **Accepted**: {report.get('total_accepted')} / "
                 f"{report.get('total_requested')} requested\n")
        if report.get("warnings"):
            body += "- **Warnings**: " + "; ".join(report["warnings"][:4]) + "\n"
    return body


@mcp.tool()
def get_quiz(quiz_id: str) -> str:
    """Fetch a stored quiz by id: its questions, answers, counts, and build
    report. Use after build_quiz to show the user the result.

    Args:
      quiz_id: the quiz id from build_quiz.
    """
    import quiz_store
    with _muted_stdout():
        quiz = quiz_store.get_quiz(quiz_id)
    if not quiz:
        return f"(No quiz found for id {quiz_id!r}.)"
    questions = quiz.get("questions") or []
    lines = [f"# Quiz: {quiz.get('title')}  (set: {quiz.get('set_name')})",
             f"- **id**: `{quiz.get('id')}` · **questions**: {len(questions)}"
             f" · **status**: {quiz.get('status')}", ""]
    for i, qu in enumerate(questions, 1):
        lines.append(f"**{i}. {qu.get('q')}** ({qu.get('type')}, "
                     f"{qu.get('difficulty')})")
        if qu.get("choices"):
            lines.append("  " + " | ".join(qu.get("choices")))
        lines.append(f"  *Answer: {qu.get('answer')}*")
        prov = qu.get("provenance") or {}
        if prov.get("title"):
            lines.append(f"  *Source: {prov.get('title')}"
                         + (f" — {prov.get('section_title') or qu.get('section_title')}"
                            if prov.get('section_title') or qu.get('section_title') else "")
                         + "*")
        lines.append("")
    report = quiz.get("report") or {}
    if report and report.get("warnings"):
        lines.append("Warnings: " + "; ".join(report["warnings"][:4]))
    return "\n".join(lines) or f"(Quiz {quiz_id} has no questions yet.)"


@mcp.tool()
def grade_answer(quiz_id: str, qid: str, response: str) -> str:
    """Grade a single answer to one question of a stored quiz. Returns
    correct/incorrect + feedback. Deterministic for mcq/true_false/fill_blank;
    LLM-assisted for short answers.

    Args:
      quiz_id: the quiz id.
      qid: the question id within the quiz.
      response: the user's answer.
    """
    agent = _load_agent()
    cfg = agent.load_config()
    import quiz_store
    import quiz_grade
    with _muted_stdout():
        quiz = quiz_store.get_quiz(quiz_id)
        if not quiz:
            return f"(No quiz found for id {quiz_id!r}.)"
        q = next((x for x in (quiz.get("questions") or [])
                  if str(x.get("qid")) == str(qid)), None)
        if q is None:
            return f"(No question {qid!r} in quiz {quiz_id!r}.)"
        graded = quiz_grade.grade_answer(q, response, cfg, agent.setup_client(cfg))
    if graded is None:
        return f"(Could not grade question {qid!r}.)"
    verdict = "correct" if graded.get("correct") else "incorrect"
    body = (f"# Grade: {verdict}\n\n- **Your answer**: {response}\n"
            f"- **Expected**: {graded.get('expected')}\n"
            f"- **Score**: {graded.get('score')}\n")
    if graded.get("feedback"):
        body += f"- **Feedback**: {graded.get('feedback')}\n"
    return body


@mcp.tool()
def review_queue(set_name: str = "", limit: int = 20) -> str:
    """Fetch the due FSRS spaced-repetition review cards (missed/weak
    questions) for a collection. Use when the user asks about their reviews
    or what to study next.

    Args:
      set_name: which index collection. Leave empty for the default.
      limit: max cards to return (1-50).
    """
    set_name = set_name or default_set_name()
    import quiz_store
    with _muted_stdout():
        cards = quiz_store.fetch_due_reviews(
            set_name=set_name, limit=max(1, min(int(limit or 20), 50)))
    if not cards:
        return f"(No review cards are due in set '{set_name}' right now.)"
    lines = [f"# Due review cards ({len(cards)})  (set: {set_name})", ""]
    for c in cards:
        q = c.get("question") or {}
        prov = q.get("provenance") or {}
        src = prov.get("section_title") or prov.get("title") or ""
        lines.append(f"- **{c.get('key')}** — {q.get('q') or ''}"
                     + (f"  (source: {src})" if src else ""))
    return "\n".join(lines)


@mcp.tool()
def get_config(keys: list = None) -> str:
    """Read current app config values. Useful to see retrieval / quiz tuning.
    Returns a compact key: value list; pass `keys` to narrow to specific keys.

    Args:
      keys: optional list of config keys to read; omit for the known keys.
    """
    agent = _load_agent()
    cfg = agent.load_config()
    if keys:
        return "\n".join(f"{k}: {cfg.get(k)}" for k in keys if k in cfg)
    notable = ["llm_base_url", "llm_model", "embed_model", "chat_mode",
               "agentic_enabled", "rerank_enabled", "lexical_backend",
               "chunking_strategy", "fsrs_enabled", "quiz_default_count",
               "quiz_batch_parents", "quiz_verify_pass", "quiz_grounding_ratio",
               "quiz_output_format", "quiz_material_chunks", "quiz_material_words",
               "quiz_sample_children"]
    return "\n".join(f"{k}: {cfg.get(k)}" for k in notable)


@mcp.tool()
def set_config(key: str, value) -> str:
    """Toggle/tweak a config value through the SESSION 1 config registry
    (only editable keys are accepted). Persists to config.local.json so the
    running web app honors it. Returns the new value.

    Args:
      key: a config key from the registry (e.g. 'rerank_enabled', 'fsrs_enabled',
        'quiz_batch_parents').
      value: the new value (bool/int/float/string; JSON string parsed properly).
    """
    editable = _editable_config_keys()
    if key not in editable:
        return f"(config key '{key}' is not writable.)"
    import json as _json
    parsed = value
    if isinstance(value, str):
        try:
            parsed = _json.loads(value)
        except Exception:
            parsed = value
        if str(parsed).lower() in ("true", "false"):
            parsed = str(parsed).lower() == "true"
    _write_local_cfg(key, parsed)
    return f"Set {key} = {parsed}"


def _editable_config_keys():
    # Derive the writable key set without importing server.py (avoid heavy deps:
    # server.py pulls Flask/chroma). server.CONFIG_META is the canonical source;
    # this mirrors the editable keys so a CLI/MCP client can tweak them. Keep in
    # sync with server.CONFIG_META (SESSION 1 registry).
    return {
        "rerank_enabled", "fsrs_enabled", "quiz_review_limit",
        "quiz_verify_pass", "quiz_grade_llm", "quiz_batch_parents",
        "quiz_parent_word_cap", "quiz_max_batches", "quiz_grounding_ratio",
        "quiz_dedupe_jaccard", "llm_structured_output", "chat_mode",
        "show_thinking", "agentic_enabled", "agentic_max_steps",
        "quiz_default_count", "quiz_output_format", "quiz_depth",
        "quiz_topic_max_works", "quiz_topic_section_pool",
        "quiz_material_chunks", "quiz_material_words",
        "quiz_sample_children",
    }


def _write_local_cfg(key, value):
    import json as _json
    from _paths import rag_root
    p = rag_root() / "config.local.json"
    data = {}
    if p.exists():
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[key] = value
    p.write_text(_json.dumps(data, indent=2), encoding="utf-8")


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

    matched = agent._match_titles(title, set_name, limit=4)
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
