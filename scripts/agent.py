#!/usr/bin/env python3
"""agent.py - CLI chat agent for RAG with fiction-aware retrieval."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for embeddings

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from _paths import merge_local_config, rag_root

console = Console()

SYSTEM_PROMPT = """You are a research assistant with access to a large personal library index. Retrieved document excerpts are your PRIMARY evidence, but you are EXPECTED to also draw on your own knowledge to fill gaps, make inferences, and connect related ideas.

RULES:
- Cited library passages are primary; cite them by title/source when you use them.
- You MAY answer beyond the excerpts using your own knowledge. LABEL the provenance of claims:
  [LIBRARY]  — directly supported by a retrieved excerpt
  [KNOWLEDGE] — from your own general knowledge, inference, or a reasonable connection
- Fiction chunks (kind="fiction") are fiction: reference them only as fiction, never as fact.
- If retrieval found no strong match (a low-relevance note is present), say plainly that the library has no direct coverage of the topic, then answer from your own knowledge clearly labeled [KNOWLEDGE], and propose 2-3 related search phrases (authors, keywords, titles) the user could try in their library.
- When coverage is thin, still suggest related topic/author/keyword routes to redirect the user.
- Never fabricate citations to passages that are not in the retrieved excerpts.
- Be concise and direct; separate what the library says from what you infer."""


def load_config():
    cfg_path = rag_root() / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    return merge_local_config(cfg)


def setup_client(cfg: dict) -> OpenAI:
    # Local servers (LM Studio / llama.cpp) accept a dummy key; a configured
    # llm_api_key enables remote/cloud OpenAI-compatible providers.
    api_key = cfg.get("llm_api_key") or "lm-studio"
    return OpenAI(
        base_url=cfg["llm_base_url"],
        api_key=api_key,
    )


def setup_chroma(set_name: str):
    client = chromadb.PersistentClient(path=str(rag_root() / "index"))
    try:
        return client.get_collection(set_name)
    except Exception:
        console.print(f"[red]Collection '{set_name}' not found. Run ingest.py first.[/red]")
        sys.exit(1)


def setup_embedder(cfg: dict):
    import torch
    from sentence_transformers import SentenceTransformer
    console.print(f"[dim]Loading embedder: {cfg['embed_model']}...[/dim]")
    torch.set_num_threads(os.cpu_count() or 8)
    model = SentenceTransformer(cfg["embed_model"], device=cfg.get("embed_device", "cpu"))
    if hasattr(model, "prompts") and "query" not in model.prompts:
        model.prompts.update({"passage": "passage: ", "query": "query: "})
    console.print("[dim]Ready.[/dim]")
    return model


# Max chunks kept per distinct title during retrieval diversification, so one
# dense book / cluster (e.g. NLP megabooks) can't monopolize the context.
MAX_CHUNKS_PER_TITLE = 3


def diversify_hits(hits: list, limit: int | None = None,
                   max_per_title: int = MAX_CHUNKS_PER_TITLE) -> list:
    """Trim a larger retrieval pool down to ``limit`` hits, preserving best
    scores but:
    - dropping chunks whose text is byte-identical to another chunk of the same
      title (duplicate ingestions / re-ingests of the same book), and
    - capping chunks per distinct title.

    Crucially, DISTINCT chunks of the same book are kept (up to the per-title
    cap) — only true duplicate text is dropped — so a multi-chunk book can be
    summarized, while repeated-by-ingestion copies and dense topic clusters
    cannot monopolize the context / source list. Input should already be sorted
    best first; the best-first order is preserved.
    """
    if limit is None:
        limit = len(hits)
    if max_per_title is None:
        max_per_title = MAX_CHUNKS_PER_TITLE
    kept, per_title, seen_text = [], {}, set()
    for h in hits:
        m = h.get("metadata") or {}
        title = m.get("title", "") or m.get("source", "") or h.get("id", "")
        if per_title.get(title, 0) >= max_per_title:
            continue
        doc = h.get("document", "")
        try:
            tkey = hashlib.sha256(doc.encode("utf-8", "replace")).hexdigest() if doc else h.get("id", "")
        except Exception:
            tkey = h.get("id", "")
        dupkey = (title, tkey)
        if dupkey in seen_text:
            continue
        seen_text.add(dupkey)
        per_title[title] = per_title.get(title, 0) + 1
        kept.append(h)
        if len(kept) >= limit:
            break
    return kept


def retrieve(query: str, embedder, collection, top_k: int = 10,
             filter_kind: str = None, cfg: dict = None, where_extra: dict | None = None):
    """Embed query and retrieve from ChromaDB.

    ``where_extra`` merges into the metadata filter, e.g. to confine retrieval
    to a specific set of titles: {"title": {"$in": ["Title A", "Title B"]}}.
    """
    query_emb = embedder.encode([query], prompt_name="query")[0].tolist()

    where = {}
    if filter_kind:
        where["kind"] = filter_kind
    if where_extra:
        where.update(where_extra)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    return hits


_TERM_STOP = set(
    "the a an and or of to in on for with by from at as is are be was were it its his her "
    "they we you your my their who what which when where how why this that these those "
    "not no can could would should will just about".split()
)

try:
    from snowballstemmer import stemmer as _snowball
    _STEMMER = _snowball("english")
except Exception:
    _STEMMER = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem_tokens(text: str) -> list[str]:
    """Lowercase + tokenize + English-stem a chunk of text.

    Used for BM25 tokenization and for the low-relevance presence check, so
    morphology variants all collapse to one stem ("horses" == "horse",
    "training" == "train", "learning" == "learn").
    """
    if not text:
        return []
    words = _TOKEN_RE.findall(text.lower())
    if _STEMMER is not None:
        return [_STEMMER.stemWord(w) for w in words]
    return words


def tokenize(text: str) -> list[str]:
    """Fast lowercase tokenization for the BM25 index build (no stemming —
    the 185k-chunk corpus would take ~12min to stem). Query tokens use the
    same scheme so they align with the index."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


_QUERY_STOP = set(
    "the a an and or of to in on for with by from at as is are be was were it its his her "
    "they their them what which whose when where how why this that these those book books "
    "summarize summary synopsis overview about tell explain give best all any some my your "
    "me please can you does do would should".split()
)

_WORK_DIRECTED = ("summarize", "summary", "synopsis", "overview", "recap",
                  "about", "spoiler", "review", "contents", "chapters", "compare")


def _is_work_directed(query: str, q: str) -> bool:
    if any(w in q.split() for w in _WORK_DIRECTED):
        return True
    return bool(re.search(r"\b(the|this)\s+\w+\s+(book|novel|series)$", query.lower()))


def _match_titles(query: str, set_name: str, limit: int = 4) -> list[str]:
    """Match a query against the known book titles in this set (manifest.db).

    Enables "specific work" retrieval: naming one work (or a small cluster of
    works) routes the query straight to those books' chunks instead of letting
    the weak embedder scatter across the whole collection.
    Returns matched titles, best first.
    """
    import sqlite3
    db = rag_root() / "manifest.db"
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT DISTINCT title FROM files WHERE set_name=? AND title IS NOT NULL AND title<>''",
            (set_name,)).fetchall()
        con.close()
    except Exception as e:
        print(f"[chat] manifest title lookup failed: {e}")
        return []

    q = re.sub(r"[^a-z0-9 ]", " ", query.lower()).strip()
    qwords = {w for w in q.split() if w not in _QUERY_STOP}
    work_directed = _is_work_directed(query, q)
    scored = []
    for (title,) in rows:
        t = re.sub(r"[^a-z0-9 ]", " ", title.lower()).strip()
        if len(t) < 3:
            continue
        twords = {w for w in t.split() if w not in _QUERY_STOP}
        if t in q or q in t:
            score = 100 + len(t)
        elif not work_directed:
            continue
        elif len(twords) >= 2 and len(twords & qwords) >= 2:
            score = 60 + len(t)
        elif len(twords & qwords) == 1 and len((twords & qwords).pop()) >= 5:
            score = 40 + len(t)
        else:
            continue
        scored.append((score, title))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, t in scored[:limit]]


def extract_terms(query: str) -> list[str]:
    """Extract distinctive terms from a query for the low-relevance guard.

    Returns (a) capitalized multi-word phrases (proper nouns) and (b) individual
    stopword-filtered words with apostrophe-s normalized (e.g. "anderson's" ->
    "anderson"), so lowercase names are still caught. If any term never appears
    in a retrieved chunk, retrieval is probably off-target.
    """
    raw = re.findall(r"[A-Za-z][A-Za-z\-']*", query)
    if not raw:
        return []

    terms = []

    # (a) capitalized phrases — runs of words starting with a capital letter
    caps, cur = [], []
    for t in raw:
        if t[0].isupper():
            cur.append(t)
        elif cur:
            caps.append(" ".join(cur))
            cur = []
    if cur:
        caps.append(" ".join(cur))
    for p in caps:
        words = [w for w in re.split(r"[^A-Za-z]+", p) if w.lower() not in _TERM_STOP]
        if words:
            terms.append(" ".join(words))

    # (b) individual distinctive words (handles lowercase names like "clinton")
    for t in raw:
        w = t.lower().rstrip("'s")
        if len(w) >= 2 and w not in _TERM_STOP:
            terms.append(w)

    # dedupe, keep phrases first
    seen, out = set(), []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


_GENERIC_WORDS = {
    "book", "books", "author", "authors", "about", "explain", "describe",
    "summarize", "summary", "meaning", "means", "difference", "compared",
    "comparison", "example", "examples", "character", "characters", "novel",
    "novels", "story", "stories", "main", "philosophy", "training", "method",
    "methods", "technique", "techniques", "write", "wrote", "ways", "way",
    "things", "thing", "best", "good", "great", "features", "feature",
}


def detect_low_relevance(query: str, hits: list, cfg: dict | None = None,
                         common_terms: set | None = None) -> tuple[bool, str]:
    """Return (low_relevance, reason) using escalating signals, all
    morphology-aware (stemmed):

    1. No hits at all.
    2. A proper-noun phrase from the query is missing (stemmed) from every
       retrieved chunk's text/title/source/tags — decisive off-target signal.
    3. A missing distinctive (non-generic) query term that is NOT a common
       corpus word is decisive — regardless of embedding score, so a high dense
       score on a semantically-near-but-wrong book (e.g. the NLP-cluster
       failure) cannot mask it. ``common_terms`` is the set of high-doc-
       frequency corpus tokens; a missing common word is tolerated because it
       may be a synonym-paraphrase (e.g. "collect" dressage jargon vs horse
       books that use "training"/"shoulder-in").
    4. Score floor: top hit below ``relevance_threshold`` (default 0.80).
    When hits carry no embedding distance (BM25-only results) the term-presence
    check alone decides.
    """
    cf = cfg or {}
    thresh = float(cf.get("relevance_threshold", 0.80))

    if not hits:
        return True, "no matching documents retrieved"

    texts = []
    for h in hits:
        texts.append(h.get("document", ""))
        m = h.get("metadata") or {}
        texts.append(m.get("title", ""))
        texts.append(m.get("source", ""))
        texts.append(str(m.get("tags", "")))
        texts.append(str(m.get("author", "")))
    corpus_stems = set(stem_tokens(" ".join(texts)))

    terms = extract_terms(query)
    for p in [t for t in terms if " " in t]:
        pstems = set(stem_tokens(p))
        if pstems and not pstems.issubset(corpus_stems):
            return True, "proper terms not found in retrieved sources: " + p

    missing = [t for t in terms
               if " " not in t and len(t) >= 3
               and t not in _GENERIC_WORDS
               and not (set(stem_tokens(t)) & corpus_stems)]
    decisive = [t for t in missing
                if not (common_terms and t in common_terms)]

    top_score = (1 - hits[0]["distance"]) if hits[0].get("distance") is not None else None
    if top_score is not None and top_score < thresh:
        return True, f"top hit score {top_score:.3f} below threshold {thresh}"

    if decisive:
        return True, ("query terms not found in retrieved sources: "
                      + ", ".join(decisive[:3]))
    return False, ""


def _llm_subqueries(query: str, context: str, client, cfg: dict) -> list[str]:
    """Ask the LLM for 2-3 follow-up search phrases to pull related/associated
    material from the index. Returns an empty list on any failure."""
    model = cfg.get("llm_model", "default")
    system = ("You expand research queries for a local e-book library search engine. "
              "Given a question and the currently retrieved context, output 2-3 SHORT "
              "search phrases (author names, keywords, related topics) most likely to "
              "surface closely related or associated material already in the library. "
              "One phrase per line. No numbering, no bullets, no explanation.")
    user = f"Question: {query}\n\nCurrently retrieved context:\n{context[:1400]}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=140,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[chat] subquery generation failed: {e}")
        return []

    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^[\s\d.\-*•)""]+", "", ln).strip()
        if len(ln) >= 3 and ln.lower() not in ("none", "n/a"):
            out.append(ln)
        if len(out) >= 3:
            break
    return out


def rerank_hits(hits: list, query: str, reranker, top_n: int = 24) -> tuple[list, bool]:
    """Re-score candidates with a cross-encoder reranker.

    Mutates ``hit["rerank_score"]`` on each dict. Returns (reranked_hits, ok)
    where handlers that fail (or a missing reranker) leave order unchanged.
    """
    if reranker is None or not hits:
        return hits, False
    pairs = [(query, (h.get("document") or "")[:1200]) for h in hits]
    try:
        scores = [float(s) for s in reranker.predict(pairs, show_progress_bar=False)]
    except Exception as e:
        print(f"[chat] rerank failed: {e}")
        return hits, False
    for h, s in zip(hits, scores):
        h["rerank_score"] = s
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order][:top_n], True


def retrieve_multi_hop(query: str, embedder, collection, client=None, cfg: dict | None = None,
                       top_k: int = 8, filter_kind: str | None = None) -> list:
    """LLM-driven multi-hop retrieval.

    Round 1: normal top-k retrieval. Then, for each additional hop, ask the LLM
    for follow-up search phrases and retrieve each, merging + deduplicating by
    chunk id. Returns hits sorted by closeness (best first).
    """
    cf = cfg or {}
    hops = max(int(cf.get("max_retrieval_hops", 2) or 1), 1)
    if hops <= 1 or client is None:
        return retrieve(query, embedder, collection, top_k=top_k,
                        filter_kind=filter_kind, cfg=cfg)

    seen, merged = set(), []

    def _add(new_hits):
        for h in new_hits:
            if h["id"] not in seen:
                seen.add(h["id"])
                merged.append(h)

    round1 = retrieve(query, embedder, collection, top_k=top_k,
                      filter_kind=filter_kind, cfg=cfg)
    _add(round1)
    if not round1:
        return merged

    context = build_context(round1, max_words=140)
    for _ in range(hops - 1):
        subqueries = _llm_subqueries(query, context, client, cf)
        if not subqueries:
            break
        for sq in subqueries:
            try:
                _add(retrieve(sq, embedder, collection, top_k=top_k,
                              filter_kind=filter_kind, cfg=cfg))
            except Exception as e:
                print(f"[chat] hop retrieval failed for {sq!r}: {e}")
        context = build_context(merged[:12], max_words=140)

    merged.sort(key=lambda h: h["distance"])
    return merged


def build_context(hits: list, max_words: int = 0) -> str:
    """Build context string from retrieval hits.

    If max_words > 0, each chunk is truncated to that many words so the
    prompt stays under the LLM server's context limit.
    """
    parts = []
    for i, h in enumerate(hits):
        meta = h["metadata"]
        kind_label = "[FICTION]" if meta.get("kind") == "fiction" else "[NON-FICTION]"
        source = f"{meta.get('title', 'Unknown')} ({meta.get('source', 'unknown')})"
        page = meta.get("page", "")
        page_str = f", p.{page}" if page else ""
        doc = h["document"]
        if max_words > 0:
            words = doc.split()
            if len(words) > max_words:
                doc = " ".join(words[:max_words]) + " …[truncated]"
        parts.append(
            f"--- Source {i+1}: {source} {kind_label}{page_str} ---\n"
            f"{doc}"
        )
    return "\n\n".join(parts)


# ─── Hybrid retrieval (BM25 lexical leg + RRF fusion) ────────────────────────
#
# These are the retrieval tuning knobs that balance precision against cost.
# They are hard-coded (not in config.json) because they trade CPU/memory against
# recall, and their sweet spot is largely corpus-size dependent:
#   BM25_TOP_N   — how many BM25 hits to take as the lexical leg before fusion.
#                  More = better recall at the cost of a bigger rerank pool.
#   BM25_BATCH   — rows per Chroma pagination call while (re)building the index.
#                  Chroma caps the returned chunk count, so large corpora MUST
#                  be walked in pages; lowering this hurts build speed, not
#                  quality.
#   FUSE_RRF_K   — the RRF constant (rank score = sum 1/(k + rank)). Larger k
#                  flattens the score curve and gives the lexical leg more
#                  weight relative to the dense pool.
BM25_TOP_N = 30
BM25_BATCH = 20000
FUSE_RRF_K = 60

_bm25_cache = {}  # set_name -> {"bm25": BM25Okapi, "ids": [chunk ids], "df": {token: doc_freq}}


def _get_bm25_index(set_name, collection, backend="auto"):
    cached = _bm25_cache.get(set_name)
    if cached is not None:
        return cached

    # Try hydrating from the persistent BM25 tables in manifest.db.
    ids, tokdocs, df = [], [], {}
    if backend != "bm25_chroma":
        try:
            import sqlite3 as _sqlite3
            db_path = Path(__file__).resolve().parent.parent / "manifest.db"
            if db_path.exists():
                conn = _sqlite3.connect(str(db_path))
                tbls = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "bm25_tokens" in tbls and "bm25_df" in tbls:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM bm25_tokens WHERE set_name = ?",
                        (set_name,)
                    ).fetchone()
                    if row and row[0] > 0:
                        print(f"[chat] hydrating BM25 from disk for '{set_name}' "
                              f"({row[0]} token rows)...", flush=True)
                        for tok, df_val in conn.execute(
                            "SELECT token, doc_freq FROM bm25_df WHERE set_name = ?",
                            (set_name,)
                        ).fetchall():
                            df[tok] = df_val
                        doc_rows = conn.execute(
                            "SELECT doc_id, token, tf FROM bm25_tokens "
                            "WHERE set_name = ? ORDER BY doc_id",
                            (set_name,)
                        ).fetchall()
                        cur_id = None
                        cur_toks = []
                        for doc_id, token, tf in doc_rows:
                            if doc_id != cur_id:
                                if cur_id is not None and cur_toks:
                                    ids.append(cur_id)
                                    tokdocs.append(cur_toks)
                                cur_id = doc_id
                                cur_toks = []
                            cur_toks.extend([token] * tf)
                        if cur_id is not None and cur_toks:
                            ids.append(cur_id)
                            tokdocs.append(cur_toks)
                        conn.close()
                        if tokdocs:
                            from rank_bm25 import BM25Okapi
                            bm25 = BM25Okapi(tokdocs)
                            print(f"[chat] BM25 index ready from disk ({len(ids)} docs).",
                                  flush=True)
                            _bm25_cache[set_name] = {
                                "bm25": bm25, "ids": ids, "df": df,
                            }
                            return _bm25_cache[set_name]
                conn.close()
        except Exception as e:
            print(f"[chat] bm25 sqlite hydrate failed, falling back to Chroma: {e}",
                  flush=True)

    # Fallback: build from Chroma (original behavior).
    if backend != "bm25_sqlite":
        count = collection.count()
        print(f"[chat] building BM25 index for '{set_name}' ({count} chunks)...", flush=True)
        ids, tokdocs = [], []
        offset = 0
        df = {}
        while True:
            res = collection.get(limit=BM25_BATCH, offset=offset, include=["documents"])
            batch_ids = res.get("ids") or []
            batch_docs = res.get("documents") or []
            if not batch_ids:
                break
            for cid, txt in zip(batch_ids, batch_docs):
                toks = tokenize(txt or "")
                if toks:
                    ids.append(cid)
                    tokdocs.append(toks)
                    for tk in set(toks):
                        df[tk] = df.get(tk, 0) + 1
            offset += len(batch_ids)
            if len(batch_ids) < BM25_BATCH:
                break
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokdocs)
        print(f"[chat] BM25 index ready ({len(ids)} docs).", flush=True)
        _bm25_cache[set_name] = {"bm25": bm25, "ids": ids, "df": df}
        return _bm25_cache[set_name]

    # Neither path produced data.
    print(f"[chat] BM25 index empty for '{set_name}'.", flush=True)
    _bm25_cache[set_name] = {"bm25": None, "ids": [], "df": {}}
    return _bm25_cache[set_name]


def _bm25_leg(set_name, query, skip_ids, collection, n=BM25_TOP_N, backend="auto"):
    """Return hits for the top-n BM25 results not already in the dense pool."""
    idx = _get_bm25_index(set_name, collection, backend=backend)
    toks = tokenize(query)
    if not toks:
        return []
    scores = idx["bm25"].get_scores(toks)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    missing = [idx["ids"][i] for i in top if idx["ids"][i] not in skip_ids]
    meta_by_id, doc_by_id = {}, {}
    if missing:
        res = collection.get(ids=missing, include=["documents", "metadatas"])
        for cid, m in zip(res.get("ids") or [], res.get("metadatas") or []):
            meta_by_id[cid] = m
        for cid, d in zip(res.get("ids") or [], res.get("documents") or []):
            doc_by_id[cid] = d

    hits = []
    for i in top:
        cid = idx["ids"][i]
        if cid in skip_ids:
            continue
        hits.append({
            "id": cid,
            "document": doc_by_id.get(cid, ""),
            "metadata": meta_by_id.get(cid, {}),
            "distance": None,
            "bm25_score": float(scores[i]),
        })
    return hits


def _rrf_fuse(*ranked_lists, k=FUSE_RRF_K):
    acc = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst):
            acc[hit["id"]] = acc.get(hit["id"], 0.0) + 1.0 / (k + rank + 1)
    by_id = {}
    for lst in ranked_lists:
        for hit in lst:
            by_id.setdefault(hit["id"], hit)
    order = sorted(acc, key=acc.get, reverse=True)
    return [by_id[cid] for cid in order]


# Context-budget tuning: controls how much retrieved text actually reaches the
# LLM prompt (a key knob for fitting the local model's context window and for
# latency, since bigger contexts = slower generation).
#   CONTEXT_WORD_BUDGET — total words allowed across all chunks in one prompt.
#                         Raise it if the LLM has a large context window and
#                         needs more evidence; lower it for faster responses.
#   CHUNK_WORD_CAP     — per-chunk truncation for flat/child chunks. Parent
#                         sections are already size-capped at ingest
#                         (parent_tokens), so this only trims children.
CONTEXT_WORD_BUDGET = 1500
CHUNK_WORD_CAP = 240


# ─── Parent-child retrieval (chunking_strategy: parent_child) ────────────────
#
# Children (small chunks) are the search unit; parents (whole sections,
# stored at ingest time in manifest.db) are the generation unit. Inactive
# unless children actually carry parent_id metadata, so flat indexes and
# flat configs take the plain child path unchanged.

def _load_parent_texts(parent_ids, set_name=None):
    """Load parent section texts from the manifest.db `parents` store.

    Returns {parent_id: {text, title, source, section_title}}; missing ids
    are simply absent from the mapping. Any failure returns {} (the caller
    then falls back to child text).
    """
    ids = [p for p in dict.fromkeys(parent_ids or []) if p]
    if not ids:
        return {}
    db = Path(__file__).resolve().parent.parent / "manifest.db"
    if not db.exists():
        return {}
    out = {}
    try:
        import sqlite3
        conn = sqlite3.connect(str(db))
        for i in range(0, len(ids), 400):
            batch = ids[i:i + 400]
            q = ("SELECT parent_id, text, title, source, section_title "
                 f"FROM parents WHERE parent_id IN ({','.join('?' * len(batch))})")
            args = list(batch)
            if set_name:
                q += " AND set_name = ?"
                args.append(set_name)
            for pid, text, title, source, sec_title in conn.execute(q, args):
                out[pid] = {"text": text or "", "title": title or "",
                            "source": source or "", "section_title": sec_title or ""}
        conn.close()
    except Exception as e:
        print(f"[chat] parent store lookup failed: {e}")
        return {}
    return out


def _map_children_to_parents(hits: list, parent_texts: dict) -> list:
    """Collapse child hits to parent (section) granularity, best child first.

    - Children whose parent text exists become ONE parent hit per parent_id
      (first/best child wins; its distance / rerank_score travel with it,
      the parent's text replaces the child's as the generation unit, and the
      matched child text is kept aside for the sources snippet).
    - Children without a parent_id, or whose parent text is missing from the
      store, pass through unchanged (child-text fallback).
    """
    out, seen_parent = [], set()
    for h in hits:
        m = h.get("metadata") or {}
        pid = m.get("parent_id") or ""
        p = parent_texts.get(pid) if pid else None
        if p:
            if pid in seen_parent:
                continue  # section already represented by its best child
            seen_parent.add(pid)
            ph = dict(h)
            meta = dict(m)
            meta["section_title"] = p.get("section_title", m.get("section_title", ""))
            ph["id"] = pid
            ph["metadata"] = meta
            ph["document"] = p.get("text") or h.get("document", "")
            ph["child_document"] = h.get("document", "")
            out.append(ph)
        else:
            out.append(h)
    return out


def retrieve_rag(set_name, query, top_k, filter_kind, cfg,
                 embedder, collection, client=None, reranker=None,
                 chat_history=None, backend="auto"):
    """Run the full retrieval pipeline and return a result dict.

    With ``chunking_strategy: parent_child`` (and a parent_child-ingested
    collection) the search unit is the small child chunk but the generation
    unit is the section-level parent text: hits are collapsed to unique
    parents and the context is built from parent text.

    Returns:
        hits, sources (deduped by (title,source)), fiction_only,
        low_relevance, relevance_reason, title_mode, matched_titles,
        message_context (the raw context string), common_terms.
    """
    # Retrieval pool sizing. We retrieve a pool 3x larger than the final top_k
    # (capped at 30) so the reranker has real candidates to reorder, then
    # diversify down to top_k (or fewer). Raising top_k (config retrieval_top_k)
    # widens coverage; lowering it tightens the prompt. The pool cap protects
    # latency since every extra candidate costs a rerank call.
    top_k = min(top_k, 8)
    pool_k = min(top_k * 3, 30)

    matched_titles = _match_titles(query, set_name)
    title_mode = len(matched_titles) > 0
    where_title = {"title": {"$in": matched_titles}} if title_mode else None

    if title_mode:
        try:
            hits = retrieve(query, embedder, collection, top_k=pool_k,
                            filter_kind=filter_kind, cfg=cfg,
                            where_extra=where_title)
        except Exception as e:
            print(f"[chat] title-gated retrieval failed, falling back: {e}")
            hits = []
        if not hits:
            title_mode = False
    else:
        hits = []

    if not title_mode:
        hops = int(cfg.get("max_retrieval_hops", 1) or 1)
        if hops > 1 and cfg.get("retrieval_hops_driver", "llm") == "llm":
            try:
                hits = retrieve_multi_hop(
                    query, embedder, collection, client=client,
                    cfg=cfg, top_k=pool_k, filter_kind=filter_kind)
            except Exception as e:
                print(f"[chat] multi-hop retrieval failed, falling back to single pass: {e}")
                hits = retrieve(query, embedder, collection, top_k=pool_k,
                                filter_kind=filter_kind, cfg=cfg)
        else:
            hits = retrieve(query, embedder, collection, top_k=pool_k,
                            filter_kind=filter_kind, cfg=cfg)

        # Lexical leg: BM25 hybrid + RRF fusion over (dense pool, BM25 top-ups).
        try:
            bm25_hits = _bm25_leg(set_name, query, {h["id"] for h in hits}, collection, backend=backend)
        except Exception as e:
            print(f"[chat] bm25 leg failed: {e}")
            bm25_hits = []
        if bm25_hits:
            dense = sorted(hits, key=lambda h: h.get("distance")
                           if h.get("distance") is not None else 2.0)
            bm = sorted(bm25_hits, key=lambda h: h["bm25_score"], reverse=True)
            hits = _rrf_fuse(dense, bm)[:max(pool_k, 40)]

        # Cross-encoder rerank of fused candidates.
        if reranker is not None:
            hits, _ = rerank_hits(hits, query, reranker, top_n=24)

    # Parent-child retrieval: map the retrieved children to their section
    # parents (search unit = child, generation unit = parent). Only active
    # when the config opts in AND the children actually carry parent_id
    # metadata (i.e. the collection was ingested with parent_child) — a flat
    # index or flat config passes through unchanged.
    parent_active = False
    if (str(cfg.get("chunking_strategy") or "flat").strip().lower()
            == "parent_child" and hits):
        if any((h.get("metadata") or {}).get("parent_id") for h in hits):
            parent_texts = _load_parent_texts(
                [(h.get("metadata") or {}).get("parent_id") for h in hits],
                set_name)
            hits = _map_children_to_parents(hits, parent_texts)
            parent_active = bool(parent_texts)

    # Assess relevance against the full candidate pool (pre-diversify).
    # In parent_child mode this pool already carries parent texts, so the
    # guard sees exactly what the generation context will contain.
    common_terms = set()
    try:
        idx = _get_bm25_index(set_name, collection, backend=backend)
        n = max(idx["bm25"].corpus_size, 1)
        common_terms = {t for t, f in idx["df"].items() if f / n >= 0.01}
    except Exception as e:
        print(f"[chat] common-terms build failed: {e}")
    low_rel, low_reason = detect_low_relevance(query, hits, cfg, common_terms=common_terms)

    # Diversify: drop duplicate/over-represented chunks of one title. In
    # parent_child mode hits are parent-level already (a section counts once).
    max_per_title = 8 if title_mode else None
    hits = diversify_hits(hits, limit=top_k if not title_mode else None,
                          max_per_title=max_per_title)

    if not hits:
        context = "(No relevant documents were found in the library for this query.)"
        sources = []
        fiction_only = False
    else:
        # Word-budget trim. Parents are large (up to parent_tokens words), so
        # fewer fit — that is expected; the budget is NOT raised to
        # compensate. In parent_child mode the first hit is always kept so an
        # oversized parent can never empty the context outright.
        kept, used = [], 0
        min_keep = 1 if parent_active else 0
        for h in hits:
            w = len(h["document"].split())
            if len(kept) >= min_keep and used + w > CONTEXT_WORD_BUDGET:
                break
            kept.append(h)
            used += w
        hits = kept

        fiction_hits = [h for h in hits if h["metadata"].get("kind") == "fiction"]
        nonfiction_hits = [h for h in hits if h["metadata"].get("kind") != "fiction"]
        fiction_only = bool(fiction_hits and not nonfiction_hits)
        # Parent text is already size-capped at ingest (parent_tokens); only
        # child chunks need the per-chunk word cap.
        context = build_context(hits, max_words=0 if parent_active else CHUNK_WORD_CAP)

        sources = []
        seen_src = set()
        for h in hits:
            m = h["metadata"]
            key = (m.get("title", "?"), m.get("source", "?"))
            if key in seen_src:
                continue
            seen_src.add(key)
            score = h.get("rerank_score")
            if score is None and h.get("distance") is not None:
                score = round(1 - h["distance"], 3)
            elif score is not None:
                score = round(float(score), 3)
            sources.append({
                "title": m.get("title", "?"),
                "source": m.get("source", "?"),
                "kind": m.get("kind", "unknown"),
                "tags": m.get("tags", "[]"),
                "score": score,
                "snippet": (h.get("child_document") or h["document"])[:300],
            })

    # In title-mode, override relevance flags (the query named real works).
    if title_mode:
        low_rel, low_reason = False, ""

    return {
        "hits": hits,
        "sources": sources,
        "fiction_only": fiction_only,
        "low_relevance": low_rel,
        "relevance_reason": low_reason,
        "title_mode": title_mode,
        "matched_titles": matched_titles,
        "message_context": context,
        "common_terms": common_terms,
    }


def format_answer(answer: str, hits: list) -> str:
    """Clean up the LLM answer and append citations."""
    return answer.strip()


def cmd_sources(hits: list):
    """Display source details."""
    table = Table(title="Retrieved Sources", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3)
    table.add_column("Kind", width=12)
    table.add_column("Title")
    table.add_column("Source")
    table.add_column("Score", justify="right")

    for i, h in enumerate(hits):
        meta = h["metadata"]
        kind = meta.get("kind", "unknown")
        kind_style = "red" if kind == "fiction" else "green"
        score = f"{1 - h['distance']:.3f}"
        table.add_row(
            str(i + 1),
            f"[{kind_style}]{kind}[/{kind_style}]",
            meta.get("title", "?"),
            meta.get("source", "?")[:60],
            score,
        )
    console.print(table)


def cmd_collections():
    """List all collections."""
    rag_root = Path(__file__).resolve().parent.parent
    client = chromadb.PersistentClient(path=str(rag_root / "index"))
    for col in client.list_collections():
        console.print(f"  {col.name}: {col.count()} chunks")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG chat agent")
    parser.add_argument("--set", default="veracrypt1", help="Collection name")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval results")
    parser.add_argument("--model", help="Override LLM model name")
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg["llm_model"] = args.model

    console.print("[bold]RAG Agent[/bold] (fiction-aware)")
    console.print(f"  Collection: {args.set}")
    console.print(f"  LLM: {cfg['llm_base_url']} ({cfg['llm_model']})")
    console.print(f"  Embedder: {cfg['embed_model']}")
    console.print()

    client = setup_client(cfg)
    collection = setup_chroma(args.set)
    embedder = setup_embedder(cfg)

    console.print(f"[dim]Indexed: {collection.count()} chunks in '{args.set}'[/dim]")
    console.print("[dim]Commands: /sources, /collections, /filter [fiction|nonfiction|all], /model <name>, /quit[/dim]")
    console.print()

    filter_mode = None  # None = no filter

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not query:
            continue

        # Commands
        if query.lower() in ("/quit", "/exit", "/q"):
            break
        elif query.lower() == "/sources":
            if hits:
                cmd_sources(hits)
            else:
                console.print("[dim]No sources yet. Ask a question first.[/dim]")
            continue
        elif query.lower() == "/collections":
            cmd_collections()
            continue
        elif query.lower().startswith("/filter"):
            parts = query.split()
            if len(parts) > 1:
                mode = parts[1].lower()
                if mode in ("fiction", "f"):
                    filter_mode = "fiction"
                    console.print(f"[dim]Filter: fiction only[/dim]")
                elif mode in ("nonfiction", "nf", "non-fiction"):
                    filter_mode = "nonfiction"
                    console.print(f"[dim]Filter: nonfiction only[/dim]")
                elif mode in ("all", "none", "off"):
                    filter_mode = None
                    console.print(f"[dim]Filter: off (all)[/dim]")
                else:
                    console.print("[dim]Usage: /filter [fiction|nonfiction|all][/dim]")
            else:
                console.print(f"[dim]Current filter: {filter_mode or 'none'}[/dim]")
            continue
        elif query.lower().startswith("/model"):
            parts = query.split(maxsplit=1)
            if len(parts) > 1:
                cfg["llm_model"] = parts[1]
                console.print(f"[dim]LLM model set to: {cfg['llm_model']}[/dim]")
            else:
                console.print(f"[dim]Current model: {cfg['llm_model']}[/dim]")
            continue
        elif query.startswith("/"):
            console.print("[dim]Unknown command. /quit /sources /collections /filter /model[/dim]")
            continue

        # Retrieve
        hits = retrieve(query, embedder, collection, top_k=args.top_k,
                       filter_kind=filter_mode, cfg=cfg)

        if not hits:
            console.print("[yellow]No relevant documents found.[/yellow]\n")
            continue

        # Check for fiction
        fiction_hits = [h for h in hits if h["metadata"].get("kind") == "fiction"]
        nonfiction_hits = [h for h in hits if h["metadata"].get("kind") != "fiction"]

        context = build_context(hits)
        user_msg = f"Question: {query}\n\nRetrieved context:\n{context}"

        if fiction_hits and not nonfiction_hits:
            user_msg += "\n\nNOTE: ALL retrieved sources are FICTION. Do NOT present them as factual. State clearly that these are fiction works."

        # Generate
        try:
            response = client.chat.completions.create(
                model=cfg["llm_model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=cfg.get("llm_temperature", 0.3),
                max_tokens=cfg.get("llm_max_tokens", 2048),
            )
            answer = response.choices[0].message.content

            console.print()
            console.print(Panel(
                Markdown(answer),
                title="[bold green]Assistant[/bold green]",
                border_style="green",
            ))

            # Source summary
            src_summary = []
            for h in hits:
                meta = h["metadata"]
                tag = "[F]" if meta.get("kind") == "fiction" else "[NF]"
                src_summary.append(f"  {tag} {meta.get('title', '?')}")
            console.print(f"[dim]Sources ({len(hits)}):[/dim]")
            for s in src_summary:
                console.print(f"  [dim]{s}[/dim]")
            console.print()

        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            console.print("[dim]Is the LLM API running on the configured URL? Check the Setup tab.[/dim]\n")


if __name__ == "__main__":
    main()
