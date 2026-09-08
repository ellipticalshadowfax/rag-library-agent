#!/usr/bin/env python3
"""agent.py - CLI chat agent for RAG with fiction-aware retrieval."""

import json
import os
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

SYSTEM_PROMPT = """You are a research assistant with access to a document index. You answer questions based on retrieved document excerpts.

IMPORTANT RULES:
- If a retrieved chunk has kind="fiction", it comes from a FICTION WORK (novel, short story, etc.). You may reference it but MUST explicitly state it is fiction and must NOT present it as factual information.
- If the only relevant chunks are fiction, say so clearly: "The only relevant sources I found are fiction works. This is reference material, not factual data."
- Non-fiction chunks can be treated as factual reference material.
- Always cite your sources: include the title and source path.
- If no relevant chunks are found, say so honestly.
- Be concise and direct."""


def load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(cfg_path) as f:
        return json.load(f)


def setup_client(cfg: dict) -> OpenAI:
    return OpenAI(
        base_url=cfg["llm_base_url"],
        api_key="lm-studio",
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


def retrieve(query: str, embedder, collection, top_k: int = 10,
             filter_kind: str = None, cfg: dict = None):
    """Embed query and retrieve from ChromaDB."""
    query_emb = embedder.encode([query], prompt_name="query")[0].tolist()

    where = {}
    if filter_kind:
        where["kind"] = filter_kind

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
            console.print("[dim]Is LMStudio running? Try: ./start_lmstudio.sh[/dim]\n")


if __name__ == "__main__":
    main()
