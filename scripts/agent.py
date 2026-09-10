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
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(cfg_path) as f:
        return json.load(f)


def setup_client(cfg: dict) -> OpenAI:
    # Local servers (LM Studio / llama.cpp) accept a dummy key; a configured
    # llm_api_key enables remote/cloud OpenAI-compatible providers.
    api_key = cfg.get("llm_api_key") or "lm-studio"
    return OpenAI(
        base_url=cfg["llm_base_url"],
        api_key=api_key,
    )


def setup_chroma(set_name: str):
    rag_root = Path(__file__).resolve().parent.parent
    client = chromadb.PersistentClient(path=str(rag_root / "index"))
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
    parser.add_argument("--set", default="library", help="Collection name")
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
