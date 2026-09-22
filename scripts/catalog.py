#!/usr/bin/env python3
"""catalog.py - Deterministic catalog / list mode for the RAG library.

Passage retrieval is deliberately capped (top_k <= 8, per-title caps, context
word budget), so a query like "list books on horsemanship" can never surface
more than a handful of titles. This module adds a separate CATALOG path that
enumerates the manifest.db `files` table (one row per file: title, kind, tags,
rel_path, set_name) and renders a deterministic Markdown table — no LLM
involved.

Public surface:
  - list_books(set_name)                 -> distinct book records, cached
  - find_books(topic, set_name, ...)     -> ranked books for a topic (3-leg fuse)
  - render_table(books)                  -> GFM Markdown table
  - list_sections(title, set_name)       -> ordered sections of one work
  - is_list_intent / is_quiz_intent      -> query intent detection
  - extract_topic / extract_filter_kind  -> query parsing helpers
  - extract_count                        -> quiz question-count parsing

All heavy imports (chromadb, agent) are deferred so this module can be
imported cheaply by the web server and the MCP server.
"""

import json
import re
import sqlite3
import sys
import threading

from _paths import rag_root

# ── Intent detection ─────────────────────────────────────────────────────────

# Primary template: "list/show/... books ... about/on <topic>".
LIST_INTENT_RE = re.compile(
    r"^(list|show|find|tell me about|how many|what books|which books|"
    r"enumerate|browse|inventory)(?:\s+of\s+(all|every|\d+)?)?"
    r"\s*(books?|novels?|works?|titles?)?\s*"
    r"(in(?:\s+my)?\s+(library|collection))?\s*"
    r"(about|on|regarding|concerning|covering)\s+",
    re.IGNORECASE,
)

# Secondary: full-catalog requests that name no topic ("list my books",
# "what books do I have", "what's in my library"). Deliberately narrow so plain
# QA questions that merely contain the word "book" ("What is the main argument
# of the book?") are NOT hijacked into a catalog listing. Only plurality-
# clarified enumeration targets count: plural "books/novels/works/titles" or
# "library/collection". A singular "the book" / "this book" / "that work"
# refers to one specific work, not a catalog.
_FULL_CATALOG_RE = re.compile(
    r"^(?:list|show|display|enumerate|browse|inventory|find|give(?: me)?|"
    r"see|what(?: are)?|which|how many)\b"
    r".*(?:books|novels|works|titles|library|collection)\b",
    re.IGNORECASE,
)

QUIZ_INTENT_RE = re.compile(
    r"(quiz|test|flashcards|exam|study guide|practice questions|"
    r"generate questions|create a quiz)",
    re.IGNORECASE,
)

_SUMMARY_KEYWORDS = {"summarize", "summary", "describe", "with notes", "overview"}

_FILTER_NONFICTION_RE = re.compile(r"\bnon[-\s]?fiction\b", re.IGNORECASE)
_FILTER_FICTION_RE = re.compile(r"\bfiction\b", re.IGNORECASE)

_TOPIC_TRIM_RE = re.compile(
    r"\s+in\s+(?:my\s+)?(?:library|collection).*$", re.IGNORECASE)
_TOPIC_LEAD_RE = re.compile(
    r"^(?:all|every|some|any|the)\s+", re.IGNORECASE)
_TOPIC_NOUN_RE = re.compile(
    r"^(?:books?|novels?|works?|titles?)\s+(?:on|about|regarding|concerning|covering)\s+",
    re.IGNORECASE)


def is_quiz_intent(query: str) -> bool:
    """Detect if the query is asking for quiz/test generation."""
    return bool(QUIZ_INTENT_RE.search(query or ""))


def is_list_intent(query: str) -> bool:
    """Detect if the query is asking for a catalog/list response.

    Quiz requests take priority and are never treated as list requests.
    """
    q = (query or "").strip()
    if not q or is_quiz_intent(q):
        return False
    if LIST_INTENT_RE.match(q):
        return True
    return bool(_FULL_CATALOG_RE.search(q))


def has_summary_keywords(query: str) -> bool:
    """Whether a list request also asks for prose (summaries) alongside it."""
    ql = (query or "").lower()
    return any(k in ql for k in _SUMMARY_KEYWORDS)


def extract_topic(query: str) -> str:
    """Extract the topic phrase from a list query.

    "what are the books in my library about training horses" -> "training horses"
    "list all my books" -> "" (empty topic means a full-catalog listing).
    """
    q = (query or "").strip().rstrip("?.! ")
    m = re.search(r"\b(?:about|on|regarding|concerning|covering)\s+(.+)$", q,
                  re.IGNORECASE)
    if not m:
        return ""
    topic = m.group(1).strip()
    topic = _TOPIC_TRIM_RE.sub("", topic)
    topic = _TOPIC_LEAD_RE.sub("", topic)
    topic = re.sub(r"^(?:books?|novels?|works?|titles?)\s+", "", topic,
                   flags=re.IGNORECASE)
    return topic.strip(" .?!")


def extract_filter_kind(query: str):
    """Extract a fiction/nonfiction filter from query text (overrides the UI)."""
    q = query or ""
    if _FILTER_NONFICTION_RE.search(q):
        return "nonfiction"
    if _FILTER_FICTION_RE.search(q):
        return "fiction"
    return None


def extract_count(query: str, default: int) -> int:
    """Extract a question count from a quiz request ("quiz me with 15 ...")."""
    m = re.search(r"\b(\d{1,3})\b", query or "")
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 100:
                return n
        except ValueError:
            pass
    return default


# ── Book enumeration ─────────────────────────────────────────────────────────

_lock = threading.RLock()
_book_cache = {}          # set_name -> {"count": int, "books": [dict]}
_chroma = {"client": None}


def _get_collection(set_name, collection=None):
    if collection is not None:
        return collection
    import chromadb
    with _lock:
        if _chroma["client"] is None:
            _chroma["client"] = chromadb.PersistentClient(
                path=str(rag_root() / "index"))
        return _chroma["client"].get_collection(set_name)


def _parse_tags(raw) -> list:
    """Normalize a tags value (JSON string / list / comma string) to a list."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(t) for t in raw if str(t).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(t) for t in parsed if str(t).strip()]
            if isinstance(parsed, str):
                return [parsed.strip()] if parsed.strip() else []
        except Exception:
            pass
        return [t.strip() for t in s.split(",") if t.strip()]
    return [str(raw)]


def _norm_kind(kind) -> str:
    return "fiction" if str(kind or "").strip().lower() == "fiction" else "nonfiction"


def _books_from_manifest(set_name: str) -> list:
    db = rag_root() / "manifest.db"
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(str(db))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT title, kind, tags, rel_path FROM files "
            "WHERE set_name=? AND title IS NOT NULL AND title<>'' "
            "ORDER BY title COLLATE NOCASE", (set_name,)).fetchall()
        con.close()
    except Exception as e:
        print(f"[catalog] manifest book lookup failed: {e}", flush=True)
        return []
    return [{
        "title": r["title"] or "",
        "kind": _norm_kind(r["kind"]),
        "tags": _parse_tags(r["tags"]),
        "source": r["rel_path"] or "",
        "set_name": set_name,
    } for r in rows]


def _books_from_chroma(collection, set_name: str) -> list:
    """Fallback: paginate Chroma metadata and dedupe by (title, source)."""
    try:
        count = collection.count()
    except Exception:
        return []
    seen, books = {}, []
    batch = 20000
    offset = 0
    while offset < count:
        try:
            res = collection.get(limit=batch, offset=offset,
                                 include=["metadatas"])
        except Exception as e:
            print(f"[catalog] chroma book scan failed: {e}", flush=True)
            break
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        if not ids:
            break
        for m in metas:
            m = m or {}
            title = m.get("title") or ""
            if not title:
                continue
            source = m.get("source") or ""
            key = (title, source)
            if key in seen:
                continue
            seen[key] = True
            books.append({
                "title": title,
                "kind": _norm_kind(m.get("kind")),
                "tags": _parse_tags(m.get("tags")),
                "source": source,
                "set_name": set_name,
            })
        offset += len(ids)
        if len(ids) < batch:
            break
    books.sort(key=lambda b: b["title"].lower())
    return books


def _dedupe(books: list) -> list:
    seen, out = set(), []
    for b in books:
        key = (b.get("title", ""), b.get("source", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def list_books(set_name: str, collection=None) -> list:
    """Return distinct book records for a set, best-first / cached.

    Prefers manifest.db `files` (one row per file); falls back to paginated
    Chroma metadata dedupe. The cache is invalidated whenever the collection
    count changes (catches re-ingests). Thread-safe.
    """
    coll = _get_collection(set_name, collection)
    try:
        count = coll.count()
    except Exception:
        count = -1
    with _lock:
        cached = _book_cache.get(set_name)
        if cached and cached["count"] == count:
            return cached["books"]

    books = _books_from_manifest(set_name)
    if not books:
        books = _books_from_chroma(coll, set_name)
    books = _dedupe(books)
    with _lock:
        _book_cache[set_name] = {"count": count, "books": books}
    return books


# ── Topic ranking (three-leg reciprocal-rank fusion) ─────────────────────────

def _rrf_titles(*ranked_lists, k: int = 60) -> list:
    acc = {}
    for lst in ranked_lists:
        for rank, title in enumerate(lst):
            acc[title] = acc.get(title, 0.0) + 1.0 / (k + rank + 1)
    return sorted(acc, key=acc.get, reverse=True)


def _title_lexical_leg(topic: str, books: list) -> list:
    from agent import tokenize
    q = set(tokenize(topic))
    if not q:
        return []
    scored = []
    tl = topic.lower().strip()
    for b in books:
        text = " ".join([b.get("title", ""), " ".join(b.get("tags") or []),
                         b.get("source", "")]).lower()
        toks = set(tokenize(text))
        overlap = len(q & toks)
        score = overlap + (3 if tl and tl in text else 0)
        if score:
            scored.append((score, b["title"]))
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    return [t for _, t in scored]


def _dense_leg(topic: str, set_name: str, cfg: dict, books: list,
               collection, embedder, filter_kind) -> list:
    from agent import retrieve
    pool = int(cfg.get("catalog_semantic_pool", 500) or 500)
    try:
        count = collection.count()
    except Exception:
        count = 0
    pool = max(1, min(pool, count))
    if pool <= 0:
        return []
    try:
        hits = retrieve(topic, embedder, collection, top_k=pool,
                        filter_kind=filter_kind, cfg=cfg)
    except Exception as e:
        print(f"[catalog] dense leg failed: {e}", flush=True)
        return []
    agg = {}
    for h in hits:
        m = h.get("metadata") or {}
        title = m.get("title") or ""
        if not title:
            continue
        dist = h.get("distance")
        score = (1 - dist) if dist is not None else 0.0
        cur = agg.get(title)
        if cur is None:
            agg[title] = [score, 1]
        else:
            cur[0] = max(cur[0], score)
            cur[1] += 1
    order = sorted(agg, key=lambda t: (agg[t][0], agg[t][1]), reverse=True)
    return order


def _bm25_leg(topic: str, set_name: str, cfg: dict, collection,
              filter_kind) -> list:
    from agent import _bm25_leg as bm25_leg
    pool = int(cfg.get("catalog_semantic_pool", 500) or 500)
    n = max(60, min(pool, 2000))
    try:
        hits = bm25_leg(set_name, topic, set(), collection, n=n,
                        backend=cfg.get("lexical_backend", "auto"))
    except Exception as e:
        print(f"[catalog] bm25 leg failed: {e}", flush=True)
        return []
    agg = {}
    for h in hits:
        m = h.get("metadata") or {}
        if filter_kind and _norm_kind(m.get("kind")) != filter_kind:
            continue
        title = m.get("title") or ""
        if not title:
            continue
        agg[title] = agg.get(title, 0.0) + float(h.get("bm25_score") or 0.0)
    return sorted(agg, key=agg.get, reverse=True)


def find_books(topic: str, set_name: str, filter_kind=None, cfg=None,
               collection=None, embedder=None, *, debug: bool = False) -> list | tuple[list, dict]:
    """Rank books by relevance to ``topic`` (empty topic = full catalog).

    Fuses three legs via reciprocal-rank weighting:
      1. title/tags/path lexical overlap
      2. aggregate dense retrieval (per-title max score + hit count)
      3. aggregate BM25 lexical retrieval
    Honours ``filter_kind``. Clamps the dense pool to collection.count().

    Args:
        topic: search topic string.
        set_name: index collection name.
        filter_kind: "fiction" / "nonfiction" / None.
        cfg: config dict (needed for semantic pool size etc.).
        collection: ChromaDB collection handle.
        embedder: sentence-transformers model handle.
        debug: when True, prints per-leg rankings to stdout and returns a
               ``(books, leg_breakdown)`` tuple so callers can inspect which
               books came from which leg and why fusion produced the final order.
               Defaults to False for normal operation.

    Returns:
        With ``debug=False`` (default): just the ranked book list.
        With ``debug=True``: ``(books_list, leg_breakdown_dict)`` where
        ``leg_breakdown`` contains per-leg top-N lists plus the fused result.
    """
    cf = cfg or {}
    books = list_books(set_name, collection)
    if filter_kind:
        books = [b for b in books if b["kind"] == filter_kind]
    if not books:
        return []

    topic = (topic or "").strip()
    if not topic:
        return sorted(books, key=lambda b: b["title"].lower())

    if collection is None:
        collection = _get_collection(set_name)
    if embedder is None:
        import agent
        embedder = agent.setup_embedder(cf)

    by_title = {b["title"]: b for b in books}
    allowed = set(by_title)

    lex_raw = _title_lexical_leg(topic, books)
    dense_raw = _dense_leg(topic, set_name, cf, books, collection,
                           embedder, filter_kind)
    bm_raw = _bm25_leg(topic, set_name, cf, collection, filter_kind)

    lex = [t for t in lex_raw if t in allowed]
    dense = [t for t in dense_raw if t in allowed]
    bm = [t for t in bm_raw if t in allowed]

    fused = _rrf_titles(lex, dense, bm)

    # If nothing ranked, fall back to the full filtered catalog so the caller
    # can still report zero matches explicitly.
    result = [by_title[t] for t in fused]

    if debug:
        # Build per-leg ranking info with position-aware output.
        # Use the raw (pre-filtered) lists for the debug display so readers
        # see what each leg actually returned before the allowed-title filter.
        leg_info: dict = {
            "query": topic,
            "set_name": set_name,
            "filter_kind": filter_kind,
            "total_titles_in_set": len(allowed),
            "lexical_top": lex[:20],
            "dense_top": dense[:20],
            "bm25_top": bm[:20],
            "fused_top": fused[:20],
        }
        # Attach numeric ranks from the raw leg outputs
        for idx, t in enumerate(lex[:20]):
            leg_info[f"lexical_rank_{idx+1}"] = t
        for idx, t in enumerate(dense[:20]):
            leg_info[f"dense_rank_{idx+1}"] = t
        for idx, t in enumerate(bm[:20]):
            leg_info[f"bm25_rank_{idx+1}"] = t
        for idx, t in enumerate(fused[:20]):
            leg_info[f"fused_rank_{idx+1}"] = t
        return result, leg_info

    return result


# ── Rendering ────────────────────────────────────────────────────────────────

def _cell(s) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_table(books: list, cfg: dict = None) -> str:
    """Render a deterministic GFM Markdown table (no LLM involved).

    Columns: #, Title, Type, Tags, Path, Kind. Tags truncate to the first 3.
    """
    max_tags = 3
    lines = ["| # | Title | Type | Tags | Path | Kind |",
             "|---|-------|------|------|------|------|"]
    for i, b in enumerate(books, 1):
        kind = b.get("kind", "unknown")
        icon = "\U0001F4D6" if kind == "fiction" else "\U0001F4DA"
        tags = b.get("tags") or []
        tag_str = ", ".join(tags[:max_tags])
        if len(tags) > max_tags:
            tag_str += f" (+{len(tags) - max_tags})"
        lines.append(
            f"| {i} | {_cell(b.get('title'))} | {icon} | {_cell(tag_str) or '—'} "
            f"| {_cell(b.get('source'))} | {_cell(kind)} |")
    return "\n".join(lines)


# ── Section enumeration (used by the get_section tool / future exams) ────────

def list_sections(title: str, set_name: str) -> list:
    """Return ordered sections/chapters of a titled work from manifest.db."""
    db = rag_root() / "manifest.db"
    if not db.exists() or not title:
        return []
    try:
        con = sqlite3.connect(str(db))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT section_title, section_ordinal, source FROM parents "
            "WHERE set_name=? AND title=? ORDER BY section_ordinal",
            (set_name, title)).fetchall()
        con.close()
    except Exception as e:
        print(f"[catalog] section lookup failed: {e}", flush=True)
        return []
    return [{"section_title": r["section_title"] or "",
             "section_ordinal": r["section_ordinal"],
             "source": r["source"] or ""} for r in rows]


# ── Deep map-reduce summaries ────────────────────────────────────────────────
#
# When the user NAMES one or more works (title_mode) and the config uses
# summary_strategy == "map_reduce", the pipeline below produces a deeper,
# book-wide summary:
#   1. retrieve a wide pool scoped to the matched titles,
#   2. map child chunks up to their section-level parents and spread them
#      evenly across the whole book by section_ordinal,
#   3. summarize the parents in small batches (map calls),
#   4. combine the sub-summaries in one final call (reduce).
# The reduce is a single LLM call, so it can be streamed token-by-token.

def _retrieve_deep_parents(set_name, matched_titles, cfg, collection, embedder):
    """Retrieve a wide pool scoped to matched titles and map to unique,
    section-ordinal-spread parents. Returns (parents, parent_texts, titles)."""
    import agent as agent_mod
    max_chunks = int(cfg.get("summary_max_chunks", 40) or 40)
    n = max(max_chunks * 3, 120)
    where = {"title": {"$in": matched_titles}}
    try:
        hits = agent_mod.retrieve("", embedder, collection, top_k=n,
                                  cfg=cfg, where_extra=where)
    except Exception:
        hits = agent_mod.retrieve(", ".join(matched_titles), embedder,
                                  collection, top_k=n, cfg=cfg,
                                  where_extra=where)
    if not hits:
        return [], {}, {}

    parent_ids = [((h.get("metadata") or {}).get("parent_id") or "")
                  for h in hits]
    parent_texts = agent_mod._load_parent_texts(parent_ids, set_name)

    parents = []
    for h in hits:
        m = h.get("metadata") or {}
        pid = m.get("parent_id") or ""
        p = parent_texts.get(pid)
        if p:
            parents.append({
                "parent_id": pid,
                "text": p.get("text") or "",
                "title": p.get("title") or m.get("title") or "",
                "section_title": p.get("section_title") or m.get("section_title") or "",
                "ordinal": m.get("section_ordinal") or 0,
            })
    seen, uniq = set(), []
    for p in parents:
        if p["parent_id"] in seen:
            continue
        seen.add(p["parent_id"])
        uniq.append(p)
    uniq.sort(key=lambda p: (p.get("ordinal") or 0, p.get("title", "").lower()))
    if len(uniq) > max_chunks:
        sample, step = [], max(len(uniq) / max_chunks, 1)
        idx = 0.0
        while len(sample) < max_chunks and int(idx) < len(uniq):
            p = uniq[int(idx)]
            if p not in sample:
                sample.append(p)
            idx += step
        uniq = sample
    titles = sorted({p.get("title", "") for p in uniq})
    return uniq, parent_texts, titles


def _map_batch_summarize(batch, book_label, cfg, client):
    """One map step: summarize a batch of parent excerpts."""
    model = cfg.get("llm_model", "default")
    parts = []
    for i, p in enumerate(batch, 1):
        sec = f" — {p.get('section_title','').strip()}" if p.get("section_title", "").strip() else ""
        parts.append(f"Excerpt {i} [{p.get('title','?')}{sec}]:\n{p.get('text','')}")
    prompt = (
        f"Summarize these excerpts from {book_label}, focusing on key arguments, "
        "structure, and notable ideas. Be thorough but concise per excerpt. "
        "Quote standout passages verbatim. Label anything directly from the "
        "text as [LIBRARY].\n\n"
        + "\n\n".join(parts))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.get("llm_temperature", 0.3),
            max_tokens=int(cfg.get("max_tokens_summary", 4096) or 4096),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(map batch failed: {e})"


def map_reduce_summary(set_name, matched_titles, cfg, collection, embedder,
                       client, progress_cb=None, stream_cb=None):
    """Run the two-pass map-reduce summary. Returns the reduced summary string.

    ``progress_cb(str)`` is called before each map batch. ``stream_cb(str)``
    receives reduce-step deltas live; when provided the reduce is streamed and
    this function returns the joined text.
    """
    parents, _pt, titles = _retrieve_deep_parents(
        set_name, matched_titles, cfg, collection, embedder)
    book_label = ", ".join(titles[:4])
    if not parents:
        return _single_shot_summary(set_name, matched_titles, cfg, collection,
                                    embedder, client)

    batch_size = 6
    batches = [parents[i:i + batch_size]
               for i in range(0, len(parents), batch_size)]
    subs = []
    for i, batch in enumerate(batches, 1):
        if progress_cb:
            progress_cb(f"Summarizing section batch {i}/{len(batches)}...")
        subs.append(_map_batch_summarize(batch, book_label, cfg, client))

    model = cfg.get("llm_model", "default")
    reduce_prompt = (
        f"Write a cohesive, well-structured summary of {book_label} by "
        "combining the section summaries below. Cover: what it's about, main "
        "arguments, structure, and notable ideas. Preserve meaningful "
        "quotations marked [LIBRARY]. Aim for depth but stay organized with "
        "headings.\n\n--- Section summaries ---\n\n"
        + "\n\n".join(f"[{i}] {s}" for i, s in enumerate(subs, 1)))
    max_tok = int(cfg.get("max_tokens_summary", 4096) or 4096)

    if stream_cb is not None:
        full = []
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": reduce_prompt}],
                temperature=cfg.get("llm_temperature", 0.3),
                max_tokens=max_tok,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                text = getattr(delta, "content", None)
                if not text and delta:
                    text = getattr(delta, "reasoning_content", None)
                if text:
                    full.append(text)
                    stream_cb(text)
        except Exception as e:
            full.append(f"\n\n(Streamed reduce failed: {e})")
        return "".join(full).strip()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": reduce_prompt}],
            temperature=cfg.get("llm_temperature", 0.3),
            max_tokens=max_tok,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Summary generation failed: {e})"


def _single_shot_summary(set_name, matched_titles, cfg, collection, embedder,
                         client):
    """Fallback: one prompt over the retrievable excerpts (no parents)."""
    import agent as agent_mod
    where = {"title": {"$in": matched_titles}}
    n = int(cfg.get("summary_max_chunks", 12) or 12)
    hits = agent_mod.retrieve(", ".join(matched_titles), embedder, collection,
                              top_k=max(n, 12), cfg=cfg, where_extra=where)
    context = agent_mod.build_context(hits[:n])
    prompt = (
        "Write a detailed summary of the work(s) named below using the "
        "retrieved excerpts. Cover what it's about, main argument, structure, "
        "and notable ideas.\n\n"
        f"Work(s): {', '.join(matched_titles)}\n\nRetrieved excerpts:\n{context}")
    model = cfg.get("llm_model", "default")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.get("llm_temperature", 0.3),
            max_tokens=int(cfg.get("max_tokens_summary", 4096) or 4096),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Summary generation failed: {e})"


# ── CLI: catalog ranking diagnostics ────────────────────────────────────────
#
# Usage:
#   python scripts/catalog.py find "<topic>" --set <name> [--debug]
#
# Without --debug: prints the normal rendered Markdown table (like the UI).
# With --debug: also prints per-leg rankings so you can see which books are
# sourced from which leg and why reciprocal-rank fusion produces the final order.
#
# The reranker (cross-encoder) is NOT wired into this catalog path — it only
# appears in /api/chat retrieval. Catalog rankings depend on the three legs
# listed below; the debug output makes those explicit.

def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Catalog ranking tool — inspect or browse book rankings")
    sub = parser.add_subparsers(dest="command")

    # --- find: topic search ---
    p_find = sub.add_parser("find", help="Rank books by topic relevance")
    p_find.add_argument("topic", help="Search topic (e.g. 'natural horsemanship')")
    p_find.add_argument("--set", dest="set_name", default="",
                        help="Index collection name (default: read from config)")
    p_find.add_argument("--filter", choices=["fiction", "nonfiction"],
                        default=None, help="Filter by fiction/nonfiction")
    p_find.add_argument("--debug", action="store_true",
                        help="Print per-leg rank breakdown (lex, dense, bm25, fused)")
    p_find.add_argument("--count", type=int, default=20,
                        help="Number of titles to show (default 20)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cfg = None
    try:
        from agent import load_config
        cfg = load_config()
    except Exception:
        pass  # will work anyway; cfg passed through find_books uses defaults

    set_name = args.set_name or ""
    if not set_name and cfg:
        # Try to pick the first available set name
        sets = cfg.get("sets", {})
        if sets:
            set_name = next(iter(sets))

    fk = args.filter  # None | "fiction" | "nonfiction"

    if args.command == "find":
        result = find_books(args.topic, set_name, filter_kind=fk, cfg=cfg,
                            debug=args.debug)

        if args.debug:
            books, leg_info = result
            print("=" * 70, flush=True)
            print(f"CATALOG RANKING DEBUG — '{args.topic}'  (set: {set_name})", flush=True)
            print(f"Total titles in set: {leg_info['total_titles_in_set']}  |  "
                  f"Filtered kind: {fk or 'all'}", flush=True)
            print("-" * 70, flush=True)

            def _print_leg(label, items, count):
                header = f"{label} (top-{min(len(items), count)})"
                print(header, flush=True)
                for idx, t in enumerate(items[:count], 1):
                    bar_len = min(len(t) // 3 + 1, 40)
                    bar = "." * bar_len
                    print(f"  {idx:>2}. {t:<{bar_len}} → {bar}", flush=True)
                print("", flush=True)

            _print_leg("Lexical (title/tags/path overlap)", leg_info["lexical_top"], args.count)
            _print_leg("Dense (E5 vector similarity, aggregated per-title)", leg_info["dense_top"], args.count)
            _print_leg("BM25 (lexical chunk matching, aggregated per-title)", leg_info["bm25_top"], args.count)
            _print_leg("FUSED (reciprocal-rank k=60, final order)", leg_info["fused_top"], args.count)

            # Show which title came from which leg(s)
            print("--- Title coverage across legs ---", flush=True)
            all_titles = set()
            all_titles.update(leg_info["lexical_top"][:args.count])
            all_titles.update(leg_info["dense_top"][:args.count])
            all_titles.update(leg_info["bm25_top"][:args.count])
            for t in sorted(all_titles):
                legs = []
                if t in leg_info["lexical_top"][:args.count]:
                    legs.append("L")
                if t in leg_info["dense_top"][:args.count]:
                    legs.append("D")
                if t in leg_info["bm25_top"][:args.count]:
                    legs.append("B")
                pos = next((i+1 for i, x in enumerate(leg_info["fused_top"][:args.count])
                            if x == t), "-")
                print(f"  {t:<50s} [{''.join(legs)}] fused_pos={pos}", flush=True)
            print("", flush=True)

            # Print the normal rendered table too
            print(render_table(books[:args.count]), flush=True)
        else:
            # debug=False → find_books returns a plain list
            _table = result if isinstance(result, list) else result[0]  # type: ignore[arg-type]
            print(render_table(_table[:args.count]), flush=True)


if __name__ == "__main__":
    _cli_main()
