#!/usr/bin/env python3
"""agent_loop.py - Bounded agentic tool-calling loop for /api/chat.

The single-shot RAG path retrieves once and generates once. This module lets
the model drive additional retrieval by calling tools mid-conversation:

  - search_library(query, top_k, filter_kind)  -> wraps agent.retrieve_rag
  - get_section(parent_id)                      -> full section text (Session 5)
  - summarize_work(title)                       -> one named work's excerpts

The loop runs the existing SYSTEM_PROMPT + retrieval context as its initial
messages, then hands the model a `tools=` list. If the model issues tool_calls,
we execute them, append the results as `tool`-role messages, and repeat until
it returns a final answer or we hit `max_steps`. Models / servers that cannot
do function-calling fall back to a ReAct text protocol: the model emits a
`call: search_library(...)` line, we parse and execute it, feed the result
back, and loop.

The loop is deliberately NON-STREAMING. The SSE path stays on the single-shot
path (see server.py).
"""

import json
import re
import time

import agent
import catalog

QUERY_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "at", "is", "are", "was", "were", "what", "which", "how",
    "book", "books", "summarize", "summary", "about", "explain",
}

_REACT_ADDENDUM = (
    "\n\nTOOL USE (fallback protocol): You may call tools to gather more "
    "material before answering. To call a tool, emit EXACTLY one line of the "
    "form `call: <tool_name>(<json args>)`, for example:\n"
    "call: search_library({\"query\": \"neuroplasticity exercises\", "
    "\"top_k\": 6})\n"
    "The tool result will be returned to you as a subsequent message starting "
    "with `Tool result:`. You may make another call, or give your final "
    "answer. When you have enough material, answer the user directly. Never "
    "invent the output of a tool you have not actually received."
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": "Search the user's e-book library index and return "
                           "the most relevant excerpts. Call to ground your "
                           "answer in the user's books.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "the question or search phrase"},
                    "top_k": {"type": "integer", "description": "1-8",
                              "default": 6},
                    "filter_kind": {"type": ["string", "null"],
                                    "description": "'fiction' or 'nonfiction'",
                                    "default": None},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "Fetch the FULL text of a book section (a parent "
                           "chunk) by its parent_id, when you need complete "
                           "context beyond a short excerpt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_id": {"type": "string",
                                  "description": "a parent_id seen in a "
                                                 "search_library excerpt"},
                },
                "required": ["parent_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_work",
            "description": "Retrieve excerpts of ONE named work (book) so you "
                           "can summarize or discuss it specifically. Set "
                           "deep=true to run a full book-wide map-reduce "
                           "summary instead of excerpt snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string",
                              "description": "the title (or distinctive part) "
                                             "of the book"},
                    "top_k": {"type": "integer", "description": "1-10",
                              "default": 8},
                    "deep": {"type": "boolean",
                             "description": "run a full map-reduce summary",
                             "default": False},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_books",
            "description": "Enumerate the user's library as a catalog table of "
                           "titles (not passage excerpts). Use this for "
                           "\"list/show/find/what books about ...\" requests. "
                           "Returns Markdown rows with title, type, tags and "
                           "path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string",
                              "description": "topic to filter by; leave empty "
                                             "to list the whole library",
                              "default": ""},
                    "filter_kind": {"type": ["string", "null"],
                                    "description": "'fiction' or 'nonfiction'",
                                    "default": None},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_quiz",
            "description": "Generate a structured practice quiz (MCQ or "
                           "short-answer) on a topic or a named work, with "
                           "collapsible answers and source citations. Useful "
                           "when the user asks to be quizzed or tested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string",
                              "description": "topic or book title to quiz on"},
                    "count": {"type": "integer",
                              "description": "number of questions",
                              "default": 10},
                    "kind": {"type": "string",
                             "description": "question type: 'mcq' or "
                                            "'short_answer'",
                             "default": "mcq"},
                },
                "required": ["topic"],
            },
        },
    },
]


def _render_hits(hits: list) -> str:
    """Format retrieved chunks as markdown grounding text (matches mcp)."""
    lines = []
    for i, h in enumerate(hits):
        meta = h["metadata"] or {}
        kind = meta.get("kind", "unknown")
        kind_label = "[FICTION]" if kind == "fiction" else "[NON-FICTION]"
        source = f"{meta.get('title', 'Unknown')} ({meta.get('source', 'unknown')})"
        page = meta.get("page", "")
        page_str = f", p.{page}" if page else ""
        score = h.get("rerank_score")
        if score is None and h.get("distance") is not None:
            score = round(1 - h["distance"], 3)
        elif score is not None:
            score = round(float(score), 3)
        score_str = f" (score {score:.3f})" if score is not None else ""
        # Surface the chapter/section heading so the model can cite the exact
        # part of the book instead of only the whole work (keeps it on-topic).
        section = meta.get("section_title") or ""
        section_str = f" — {section.strip()}" if section and section.strip() else ""
        lines.append(f"## Source {i+1} {kind_label}{score_str} — {source}{section_str}{page_str}")
        lines.append(h["document"])
        lines.append("")
    return "\n".join(lines)


def _tool_search_library(query, top_k, filter_kind, set_name, cfg,
                         embedder, collection, reranker, backend):
    top_k = max(1, min(int(top_k or 6), 8))
    rag = agent.retrieve_rag(
        set_name, query, top_k, filter_kind, cfg,
        embedder=embedder, collection=collection,
        client=None, reranker=reranker, backend=backend)
    hits = rag["hits"]
    if not hits:
        return (f"(No matching documents were retrieved from the library for "
                f"this query in set '{set_name}'.)")
    head = [f"# Library search: \"{query}\"  (set: {set_name})", ""]
    if rag["fiction_only"]:
        head.append("NOTE: ALL retrieved sources are FICTION. Do NOT present "
                    "them as fact.")
        head.append("")
    if rag["low_relevance"]:
        note = rag["relevance_reason"] or "low relevance"
        head.append("NOTE: Retrieval found no strong match "
                    f"({note}). Say the library lacks direct coverage and "
                    "answer from your own knowledge.")
        head.append("")
    return "\n".join(head) + _render_hits(hits)


def _tool_get_section(parent_id, set_name):
    texts = agent._load_parent_texts([parent_id], set_name)
    p = texts.get(parent_id)
    if not p:
        return (f"(No full section found for parent_id {parent_id!r}. It may "
                "not be a parent_child index.)")
    title = p.get("title") or "Unknown"
    sec = p.get("section_title") or ""
    head = f"# Section: {title}"
    if sec:
        head += f" — {sec}"
    return head + "\n\n" + (p.get("text") or "")


def _tool_summarize_work(title, top_k, set_name, cfg, embedder, collection,
                         reranker):
    top_k = max(1, min(int(top_k or 8), 10))
    matched = agent._match_titles(title, set_name, limit=4)
    if not matched:
        return (f"(No work matching '{title}' was found in set '{set_name}'. "
                "Try a different title or use search_library instead.)")
    where = {"title": {"$in": matched}}
    hits = agent.retrieve(title, embedder, collection,
                          top_k=min(top_k * 3, 24), cfg=cfg, where_extra=where)
    hits = agent.diversify_hits(hits, limit=None, max_per_title=top_k)
    fiction = [h for h in hits if h["metadata"].get("kind") == "fiction"]
    head = [f"# Summarize requested work(s): {', '.join(matched)}  "
            f"(set: {set_name})", ""]
    if fiction:
        head.append("NOTE: These works are FICTION. Present them as fiction, "
                    "not fact.")
        head.append("")
    return "\n".join(head) + _render_hits(hits)


def _tool_list_books(topic, filter_kind, set_name, cfg, collection, embedder):
    try:
        books = catalog.find_books(topic, set_name, filter_kind, cfg,
                                   collection=collection, embedder=embedder)
    except Exception as e:
        return f"(list_books failed: {e})"
    if not books:
        return (f"(No books matching{f' topic {topic!r}' if topic else ''} "
                f"were found in set '{set_name}'.)")
    head = (f"# Library catalog{f' — books about **{topic}**' if topic else ''}"
            f"  (set: {set_name})\n\n"
            f"Found {len(books)} matching title(s).\n\n")
    rows = catalog.render_table(books[:50])
    out = head + rows
    if len(books) > 50:
        out += f"\n\n_(truncated — {len(books) - 50} more not shown)_"
    return out


def _tool_summarize_work_deep(title, ctx):
    """Deep map-reduce summary of a named work via catalog.map_reduce_summary."""
    set_name = ctx["set_name"]
    cfg = ctx["cfg"]
    matched = agent._match_titles(title, set_name, limit=4)
    if not matched:
        return (f"(No work matching '{title}' was found in set '{set_name}'. "
                "Try a different title or use search_library instead.)")
    try:
        summary = catalog.map_reduce_summary(
            set_name, matched, cfg, ctx["collection"], ctx["embedder"],
            ctx["client"])
    except Exception as e:
        return f"(deep summarize_work failed: {e})"
    if not summary or "failed" in summary:
        return summary or "(Deep summary produced no output.)"
    return (f"# Deep summary of: {', '.join(matched[:4])}  (set: {set_name})\n\n"
            + summary)


def _tool_make_quiz(topic, count, ctx):
    """Generate a quiz on a topic/work via study.generate_quiz."""
    import study
    set_name = ctx["set_name"]
    cfg = ctx["cfg"]
    topic = (topic or "").strip()
    try:
        count = max(1, min(int(count or 10), 50))
    except (TypeError, ValueError):
        count = int(cfg.get("quiz_default_count", 10) or 10)
    import agent as agent_mod
    embedder = ctx["embedder"]
    collection = ctx["collection"]
    # Keep the material small enough to fit a tight-window model (see the note
    # on QUIZ_MATERIAL_* in server.py) — 24 × 300 words blows an 8k context.
    MC, MW = 10, 140
    material = None
    matched = agent_mod._match_titles(topic, set_name)
    if matched:
        where = {"title": {"$in": matched}}
        try:
            hits = agent_mod.retrieve(", ".join(matched), embedder, collection,
                                      top_k=MC, cfg=cfg, where_extra=where)
        except Exception:
            hits = []
        material = agent_mod.build_context(hits[:MC], max_words=MW)
    else:
        books = catalog.find_books(topic, set_name, None, cfg,
                                   collection=collection, embedder=embedder)
        titles = [b["title"] for b in books[:5]]
        if titles:
            where = {"title": {"$in": titles}}
            try:
                hits = agent_mod.retrieve(topic, embedder, collection,
                                          top_k=MC, cfg=cfg, where_extra=where)
            except Exception:
                hits = []
            material = agent_mod.build_context(hits[:MC], max_words=MW)
    if not material:
        return (f"(No material found to quiz on{f' topic {topic!r}' if topic else ''}"
                f" in set '{set_name}'.)")
    saved = dict(cfg)
    saved["quiz_default_count"] = count
    questions, err = study.generate_quiz(topic or ", ".join(matched), material,
                                         saved, ctx["client"])
    if not questions:
        return f"(Quiz generation failed: {err or 'no questions parsed'})"
    head = f"# Quiz{f' — {topic}' if topic else ''}  (set: {set_name})\n\n"
    return head + study.render_markdown(questions)


def _exec_tool(name, args, ctx):
    """Dispatch a tool call (function-calling or ReAct) to its Python impl."""
    a = args or {}
    if name == "search_library":
        return _tool_search_library(
            a.get("query", ""), a.get("top_k", 6), a.get("filter_kind"),
            ctx["set_name"], ctx["cfg"], ctx["embedder"], ctx["collection"],
            ctx["reranker"], ctx["backend"])
    if name == "get_section":
        return _tool_get_section(a.get("parent_id", ""), ctx["set_name"])
    if name == "summarize_work":
        if a.get("deep"):
            return _tool_summarize_work_deep(a.get("title", ""), ctx)
        return _tool_summarize_work(
            a.get("title", ""), a.get("top_k", 8), ctx["set_name"],
            ctx["cfg"], ctx["embedder"], ctx["collection"], ctx["reranker"])
    if name == "list_books":
        return _tool_list_books(
            a.get("topic", "") or "", a.get("filter_kind"),
            ctx["set_name"], ctx["cfg"], ctx["collection"], ctx["embedder"])
    if name == "make_quiz":
        return _tool_make_quiz(a.get("topic", ""), a.get("count", 10), ctx)
    return f"(Unknown tool: {name})"


def _build_initial_messages(set_name, query, history, cfg, embedder,
                            collection, reranker, backend, top_k, filter_kind):
    """Run the first retrieval and build system/history/user messages,
    mirroring server._prepare_rag. Returns (messages, rag_result)."""
    rag = agent.retrieve_rag(
        set_name, query, top_k, filter_kind, cfg,
        embedder=embedder, collection=collection,
        client=None, reranker=reranker, backend=backend)

    title_mode = rag["title_mode"]
    matched_titles = rag["matched_titles"]
    context = rag["message_context"]
    fiction_only = rag["fiction_only"]
    low_rel = rag["low_relevance"]
    low_reason = rag["relevance_reason"]

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

    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
    for m in history or []:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})
    return messages, rag


_REACT_CALL_RE = re.compile(
    r"call:\s*([a-zA-Z_]\w*)\s*\(\s*(.*?)\s*\)", re.IGNORECASE | re.DOTALL)


def _parse_react_call(text: str):
    """Return (name, args_dict) for the first `call: tool(...)` line, else None."""
    m = _REACT_CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    raw = m.group(2).strip()
    args = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                args = parsed
        except Exception:
            # lenient single-arg fallback: tool("some string")
            inner = raw.strip().strip('"').strip("'")
            if inner:
                args = {"query": inner}
    return name, args


def run_agent_loop(set_name, query, history, cfg, client, max_steps=None,
                   top_k=None, filter_kind=None, embedder=None, collection=None,
                   reranker=None, backend="auto", strategy=None,
                   stream_cb=None, progress_cb=None):
    """Run the bounded agentic loop. Returns a result dict:

        {"answer", "sources", "fiction_only", "low_relevance",
         "relevance_reason", "steps", "strategy"}

    When ``stream_cb`` is provided the final answer is produced by ONE extra
    streamed chat completion: each delta (with ``reasoning_content`` filtered
    out unless ``cfg.show_thinking`` is true) is passed to ``stream_cb``, and
    ``stream_cb({"__done__": True, ...})`` fires at the end. ``progress_cb``,
    when given, is called ~every 1.5s while the model is resolving tool
    calls, to keep long-running SSE connections alive. With neither callback
    the loop behaves exactly as before (blocking, non-streamed).
    """
    if top_k is None:
        top_k = int(cfg.get("retrieval_top_k", 10))
    if max_steps is None:
        max_steps = int(cfg.get("agentic_max_steps", 3) or 3)
    if strategy is None:
        strategy = str(cfg.get("agentic_strategy", "auto") or "auto").lower()

    if embedder is None:
        embedder = agent.setup_embedder(cfg)
    if collection is None:
        collection = agent.setup_chroma(set_name)

    ctx = {"set_name": set_name, "cfg": cfg, "embedder": embedder,
           "collection": collection, "reranker": reranker, "backend": backend,
           "client": client}

    messages, rag = _build_initial_messages(
        set_name, query, history, cfg, embedder, collection, reranker,
        backend, top_k, filter_kind)

    model = cfg.get("llm_model", "default")
    temperature = cfg.get("llm_temperature", 0.3)
    max_tokens = cfg.get("llm_max_tokens", 2048)
    steps = 0
    streaming = stream_cb is not None

    def _finish(answer, protocol, sources=None):
        """Return the result dict. When streaming, regenerate the final answer
        as one streamed completion appended to the accumulated messages."""
        sources = rag["sources"] if sources is None else sources
        if not streaming:
            return {
                "answer": answer, "sources": sources,
                "fiction_only": rag["fiction_only"],
                "low_relevance": rag["low_relevance"],
                "relevance_reason": rag["relevance_reason"],
                "steps": steps, "strategy": protocol,
            }
        # One extra streamed completion so the client sees live token deltas.
        show_thinking = bool(cfg.get("show_thinking", False))
        acc = []
        try:
            stream = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature,
                max_tokens=int(cfg.get("max_tokens_default",
                                       max_tokens) or max_tokens),
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                reason = getattr(delta, "reasoning_content", None)
                if show_thinking and reason:
                    text = (text or "") + reason
                if text:
                    acc.append(text)
                    stream_cb(text)
        except Exception as e:
            err = f"\n\n(Streamed final answer failed: {e})"
            acc.append(err)
            stream_cb(err)
        result = {
            "answer": "".join(acc).strip(), "sources": sources,
            "fiction_only": rag["fiction_only"],
            "low_relevance": rag["low_relevance"],
            "relevance_reason": rag["relevance_reason"],
            "steps": steps, "strategy": protocol,
        }
        stream_cb(dict(result, **{"__done__": True}))
        return result

    # Choose protocol. function-calling for "auto" or "function"; ReAct for
    # "react"; auto degrades to ReAct if a call with tools errors out.
    protocol = "react" if strategy == "react" else "function"
    last_beat = time.monotonic()

    def _heartbeat():
        nonlocal last_beat
        if not progress_cb:
            return
        now = time.monotonic()
        if now - last_beat >= 1.5:
            last_beat = now
            progress_cb("Resolving...")

    while steps < max_steps:
        steps += 1
        call_kwargs = dict(model=model, messages=messages,
                           temperature=temperature, max_tokens=max_tokens)
        if protocol == "function":
            call_kwargs["tools"] = _TOOLS
            call_kwargs["tool_choice"] = "auto"

        try:
            resp = client.chat.completions.create(**call_kwargs)
        except Exception as e:
            print(f"[agent_loop] LLM call failed (step {steps}): {e}")
            if protocol == "function":
                # Function-calling unsupported/errored -> degrade to ReAct.
                print("[agent_loop] falling back to ReAct protocol")
                protocol = "react"
                continue
            # ReAct also failed: cannot proceed. Reuse the retrieved context
            # and let the caller's retry/error handling decide.
            return {
                "answer": "", "sources": rag["sources"],
                "fiction_only": rag["fiction_only"],
                "low_relevance": rag["low_relevance"],
                "relevance_reason": rag["relevance_reason"],
                "steps": steps, "strategy": protocol,
                "error": f"LLM error: {e}",
            }

        msg = resp.choices[0].message

        if protocol == "function":
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in tool_calls],
                })
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    print(f"[agent_loop] tool call: {name}({args})")
                    _heartbeat()
                    result = _exec_tool(name, args, ctx)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": result,
                    })
                continue  # loop for the model to use the results

            # No tool_calls -> final answer.
            answer = (msg.content or "").strip()
            if not answer and getattr(msg, "reasoning_content", None):
                answer = msg.reasoning_content.strip()
            return _finish(answer, "function")

        # ── ReAct protocol ──
        text = (msg.content or "").strip()
        call = _parse_react_call(text)
        if call:
            name, args = call
            # Drop any prose that surrounds the call so the next model turn
            # sees clean context.
            stripped = _REACT_CALL_RE.sub("", text).strip()
            if stripped:
                messages.append({"role": "assistant", "content": text})
            print(f"[agent_loop] react call: {name}({args})")
            _heartbeat()
            result = _exec_tool(name, args, ctx)
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{result}\n\n"
                           "Continue: make another call or give your final answer.",
            })
            continue

        # No call line -> final answer.
        answer = text or (getattr(msg, "reasoning_content", None) or "").strip()
        return _finish(answer, "react")

    # Ran out of steps without a final answer.
    last = messages[-1].get("content", "") if messages else ""
    return _finish(last, protocol)