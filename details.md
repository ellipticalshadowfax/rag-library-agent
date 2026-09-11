# How RAG Works Under the Hood

This document walks through the **entire processing chain** of the RAG Library
Agent, from a raw ebook sitting on your disk to an answer in the chat window. It
is written for end users who want to understand the tool, and for anyone studying
how retrieval-augmented generation actually works.

If you just want to use the app, the [README](README.md) is enough. Read this when
you want to know *why* it works and what is happening at each step.

---

## The big picture

At the highest level, the system does three jobs:

1. **Indexing (offline):** turn your books into a searchable database of vectors.
2. **Retrieval (per question):** find the most relevant passages in that database.
3. **Generation (per question):** hand those passages to an LLM that writes an answer.

```
Your library ──► extract ──► chunk ──► embed ──► ChromaDB vector index
 (PDF/EPUB)         │                                      │
                    │                                      │
                    │          your question ──────────────┤
                    │                 │                    │
                    │                 ▼                    ▼
                    │          embed query ──► nearest neighbors (dense)
                    │                 │         + BM25 lexical match
                    │                 │                    │
                    │                 ▼                    ▼
                    │            fuse + rerank ──► top chunks ──► LLM ──► answer
                    └── OCR if scanned ─────────────────────────┘
```

---

## Part 1 — The embedding model (the heart of everything)

### What is an embedding?

An **embedding** is a list of numbers (a *vector*) that represents the *meaning*
of a piece of text. For example, the default model here produces vectors of
**384 dimensions** — that is, each chunk of text is turned into 384 floating-point
numbers.

The key property of a good embedding model is that **semantically similar texts
end up with numerically close vectors**. "How do I make a béarnaise sauce?" and
"recipe for hollandaise" will be closer together in this 384-dimensional space
than either is to "how to saddle a horse." This is what makes **semantic search**
possible: we don't need the user's exact words, we need the user's *meaning*.

### Which model, and why it's multilingual

The default is `intfloat/multilingual-e5-small`. The name tells you a lot:

- **`multilingual`** — it was trained on dozens of languages (English, Spanish,
  French, German, Russian, and many more). You can ask a question in one language
  and retrieve passages written in another. This is a major benefit for people
  with mixed-language libraries.
- **`e5`** — a family of models trained specifically for retrieval. E5 models
  expect a small *prompt prefix* (see below) that tells them whether text is a
  passage to index or a query to search with.
- **`small`** — a compact model (about 384-dim vectors). It is fast on a CPU and
  the whole model is only ~50 MB, but it trades some accuracy for that speed.

The model is loaded with the `sentence-transformers` library. You can swap it for
a bigger one (`bge-m3`, `Qwen3-Embedding`) in `config.json` if you want higher
quality and don't mind slower indexing.

### Why "passage:" and "query:" prefixes matter

E5-style models were trained so that **passages being indexed** are prepended with
the literal text `passage:` and **search queries** are prepended with `query:`.
Mixing these up noticeably degrades retrieval quality.

So the program is careful:

- **At ingest time** it embeds each chunk with `prompt_name="passage"` →
  `"passage: <chunk text>"`.
- **At query time** it embeds your question with `prompt_name="query"` →
  `"query: <your question>"`.

Both vectors then live in the same space, which is what makes the cosine
similarity between them meaningful.

### Where the vectors live

The vectors are stored in **ChromaDB**, an embedded vector database that lives on
disk under the `index/` directory. Chroma does the fast "nearest neighbor" search:
it stores the vectors and, given a query vector, returns the `n` vectors whose
cosine similarity to the query is highest.

---

## Part 2 — Ingest: turning books into a database

"Indexing" is the offline process that converts your files into this searchable
database. It runs from the **Ingest** tab or the CLI
(`scripts/ingest.py`), and it works in clear stages.

### 1. Text extraction

First the raw file is opened and its text is pulled out. Different formats use
different extractors (`extract_text` in `scripts/ingest.py`):

| Format | Extractor |
|--------|-----------|
| PDF | PyMuPDF (`fitz`), reading the built-in text layer |
| EPUB | `ebooklib` + a custom HTML parser that preserves paragraphs |
| MOBI | `mobi` extractor, concatenating the HTML parts |
| TXT / HTML | plain read, HTML tags stripped |
| DJVU | external `djvutxt` tool |

**Important nuance for PDFs:** some PDFs are *scanned images* with no text layer at
all. Those get routed to OCR (Part 5). PDFs with a real text layer are used
directly.

### 2. Chunking

Books are far too large to embed as one vector (the meaning of a 300-page book is
not one number, and the LLM can't read 300 pages of context). So the text is split
into **chunks** — small, self-contained passages.

The defaults are:

- `chunk_tokens: 330` — each chunk is roughly 330 tokens (≈ 250 words).
- `chunk_overlap: 60` — consecutive chunks overlap by 60 tokens, so no sentence or
  idea is chopped at a boundary and lost.
- `chunking_strategy: parent_child` — see below.

The chunker is **boundary-aware**: it prefers to split at *section → paragraph →
sentence* boundaries before falling back to a simple word count. This keeps chunks
semantically coherent instead of cutting mid-sentence.

### 3. Parent–child chunking (the clever part)

With `parent_child` chunking, every small child chunk is linked to a bigger
**parent** — a full section (~1200 words, e.g. one chapter or one EPUB spine item).
The system retrieves by **child** (small, precise, good for matching) but generates
answers from the **parent** (large, has full context).

Think of it this way: the child is the "bookmark" that points at the right part of
a book; the parent is the actual page you then read. This gives you precise
retrieval *and* complete context, rather than forcing you to pick one.

### 4. Embedding

Each chunk is passed through the embedding model with the `passage:` prefix,
producing a 384-dimensional vector. Everything runs on the **CPU** (`torch.no_grad`,
batched for speed) — no GPU needed.

### 5. Deterministic IDs and safe re-ingest

Every chunk gets a stable ID: `sha256(f"{file_path}:{chunk_index}")`. Because the
ID depends only on the file and the chunk position, **re-running ingest just
upserts the same rows in place** — it never duplicates. This is what makes
incremental re-indexing and `--force` rebuilds safe.

### 6. Batching for huge files

Chroma can only accept about 5461 vectors in one call. Books that produce
thousands of chunks (like an omnibus) are embedded and upserted in **batches of
500** (`ingest_batch_size`), so no file is ever skipped for being too big. Files
producing more than 5000 chunks are flagged as `BIG` and handled across multiple
batches.

### 7. What metadata is stored

Each chunk carries useful metadata, not just a vector:

- `source` — the relative file path (e.g. `Author/Title/file.pdf`)
- `title` — the book title (from Calibre metadata, or the filename)
- `kind` — `fiction` or `nonfiction`, derived from Calibre tags
- `tags` — the original Calibre tags
- `set` — which library set it belongs to
- `chunk_index` / `total_chunks` — position within the file
- In parent–child mode: `section_title`, `section_ordinal`, `parent_id`

A separate SQLite file, `manifest.db`, tracks which files have been indexed and
stores the **BM25 lexical index** (Part 4) and the parent texts.

---

## Part 3 — Semantic search: answering your question

When you type a question and hit send, the retrieval pipeline runs. This is the
most intricate part, and it combines several complementary techniques.

### Step 0 — Title short-circuit

Before any vector math, the system checks whether your question is really "find me
a specific book." If your question matches a known title in the library (e.g.
"summarize the Hobbit"), it retrieves only from that book and asks the LLM to
summarize that specific work. This is fast and precise.

### Step 1 — Dense retrieval (the semantic leg)

Your question is embedded with the `query:` prefix, producing a query vector. Then:

```
query_vector · all_chunk_vectors  →  cosine similarity scores
→  keep the highest-scoring chunks
```

This is the **dense** (semantic) retrieval leg. It understands meaning and
synonyms: asking "how to cook a pot roast" can match a passage that says
"braise the beef low and slow." This is the core benefit of semantic search over
old-style keyword matching — it doesn't care about exact wording, and it's
**multilingual** because the embedding model is multilingual.

The pool retrieved here is deliberately 3× larger than the final answer needs
(`top_k` × 3, capped at 30) so later stages have real candidates to choose from.

### Step 2 — BM25 lexical retrieval (the keyword leg)

Dense search has a weakness: it can miss rare, exact terms (names, numbers,
specialized jargon) because those aren't "close enough" to the query in semantic
space. To fix this, the system also runs a classic **BM25** keyword search — the
same family of algorithm that powered old search engines.

For this it builds a **lexical index**: for each word it records which chunks it
appears in and how often (a *term → document-frequency* map). It's built lazily
from `manifest.db` (so a ~186k-chunk library loads fast) and cached in memory.

BM25 scores each chunk by how many of your query's *exact* words it contains, with
rarer words weighted more heavily.

> **Two tokenizers, on purpose.** The BM25 index uses *raw* tokenization (lowercase,
> no stemming) because stemming 186k+ chunks would take ~12 minutes. A separate,
> lightweight *stemming* tokenizer is used only for the small low-relevance guard
> in Part 4.

### Step 3 — Reciprocal Rank Fusion (fusing the two legs)

Now we have two ranked lists: a dense list and a lexical list. They're merged with
**Reciprocal Rank Fusion (RRF)**:

```
score(chunk) = Σ  over each list that contains it:  1 / (k + rank + 1)
```

with `k = 60`. A chunk that ranks high in **both** lists gets a strong combined
score; a chunk that only one leg found still gets partial credit. This is the best
of both worlds: semantic understanding *plus* exact keyword precision.

### Step 4 — Cross-encoder reranking

The fused candidates are still approximate. The system then re-scores the top ones
with a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Unlike the
bi-encoder used for retrieval, a cross-encoder sees the **query and the document
together as one input** and outputs a single relevance score — much more accurate,
but too slow to run over the whole library. That's why it's only applied to the
~24 already-shortlisted candidates. (Disable with `rerank_enabled: false` if it's
too slow on your CPU.)

### Step 5 — Low-relevance guard

A warning system catches cases where retrieval found nothing relevant, so the LLM
won't confidently answer from garbage. It checks, over the *full* candidate pool:

- Are there **any** hits at all?
- Did a **proper-noun phrase** from the query (e.g. "Lord Byron") fail to appear?
- Did a **rare, distinctive term** (e.g. "spherification") fail to appear? A missing
  *common* corpus word (like "collect") is tolerated as a synonym-paraphrase, but a
  missing rare term is a red flag.
- Is the **top similarity score** below a threshold (`relevance_threshold: 0.8`)?

If any signal fires, the answer is flagged low-relevance and the LLM is told to
say so plainly and suggest better search phrases instead of hallucinating.

### Step 6 — Diversification and context trimming

Results are **deduplicated** (identical chunks dropped) and **capped per title**
so one book can't monopolize the context. In parent–child mode, multiple child hits
from the same section are **collapsed to one parent** — the best child "wins" and
its parent text is used for generation.

The final context is trimmed to a **word budget** (`CONTEXT_WORD_BUDGET: 1500`
words, each chunk capped at 240 words) so it fits in the LLM's context window.
The `sources` list you see in the UI is deduped by `(title, source)`, so a single
book appears as **one** source even if several of its chunks are in the context.

---

## Part 4 — Multi-hop searching

A single search often isn't enough. **Multi-hop retrieval** lets the system search
multiple times, using the first round of results to inform the second.

How it works (`max_retrieval_hops: 2`):

1. **Hop 1:** run the normal top-k retrieval on your question.
2. **Hop 2 (and later):** hand the question *plus the results so far* to the LLM,
   and ask it to expand the research:
   > "output 2–3 SHORT search phrases (author names, keywords, related topics)."
3. Each generated subquery is run through retrieval again, and the new results are
   **merged and deduplicated** by chunk ID with the earlier results.
4. Everything is sorted by relevance and passed forward.

So a question like "how does the thyroid affect metabolism?" might spawn follow-up
searches for "thyroid hormone function," "basal metabolic rate," and the name of a
relevant author — pulling in related passages the first search missed.

**Follow-up questions in a conversation** are handled separately but simply: prior
turns of the conversation are injected into the message history before your new
question, so the LLM has context about what was already discussed (bounded to the
last 12 messages / 6000 chars).

---

## Part 5 — OCR: reading scanned books

Many PDFs (especially older books) are just scanned images with no text. They can't
be chunked or embedded directly because there is no text to extract. That's where
**optical character recognition (OCR)** comes in.

The pipeline:

1. **Detect** whether a PDF needs OCR (no text layer, or too little text).
2. **Compare engines** (`ocr_compare.py`): sample a few pages, OCR them with **both**
   Tesseract and RapidOCR, score each result for "well-formedness" (accuracy +
   low garbage), and pick the better engine for that file.
3. **OCR the full file** with the winner, page by page, writing a text cache.
4. **Merge** (`merge_ocr_into_pdf.py`): write the recognized text back into the PDF
   as an **invisible, searchable text layer** — the words are there for copying and
   search but not drawn on the page. This makes the PDF searchable forever after.

OCR is CPU-bound and much slower than text extraction, which is why it's opt-in
(`ocr_enabled`). During ingest, if OCR is enabled and a file needs it, the OCR text
is what gets chunked and embedded.

---

## Part 6 — The LLM: generation and local model internals

### How the LLM is called

The RAG system doesn't generate from its own internal model — it delegates to a
**separate OpenAI-compatible LLM server** (`llm_base_url`, default
`http://localhost:1234/v1`, i.e. LM Studio or llama.cpp). Any OpenAI-compatible
endpoint works: local LM Studio, llama.cpp, vLLM, or a cloud API.

The request is:

```
messages = [
  {system: SYSTEM_PROMPT},
  ...conversation history...,
  {user: "Question: <your question>\n\nRetrieved context: <top chunks>"}
]
→ client.chat.completions.create(model, messages, temperature=0.3, max_tokens=4096)
```

The model is told (in the `SYSTEM_PROMPT`) that the retrieved passages are its
**primary evidence** but that it may also use its own knowledge — and, crucially,
that it must **label the provenance** of every claim:

- `[LIBRARY]` — directly supported by a retrieved excerpt
- `[KNOWLEDGE]` — from the model's own knowledge or inference

It must never cite a passage that isn't in the context, must treat fiction chunks
as fiction, and must say plainly when the library has no coverage.

### Streaming

The chat streams tokens in real time (server-sent events: `delta`/`done`/`error`),
so you see the answer build up word by word rather than waiting for the whole thing.
Each finished answer is persisted to a conversation JSON file.

### What "the LLM" actually is: weights, parameters, and MoE

The repository *doesn't run* the LLM — it talks to one you provide. But since this
document is for people studying the concept, here's what is going on inside that
separate server, because it's directly relevant to RAG:

- **Weights.** An LLM is fundamentally a giant matrix of numbers — its *weights* —
  learned during training. Generation is arithmetic over these matrices: the model
  predicts, one token at a time, the most likely next token given the context.
- **Parameters.** The "size" of a model is its number of parameters (e.g. a 1.7B
  model has ~1.7 billion weights). More parameters = more capacity, but more memory
  and compute. This is a major constraint for local use.
- **Quantization.** To fit a large model in a small amount of RAM, its weights are
  compressed — stored with fewer bits per number (e.g. `Q4_K_M` = 4-bit quantized).
  This trades a little quality for a big reduction in memory. Local deployments
  almost always run quantized models (the included optional self-hosted model,
  `Qwen3-1.7B-Q4_K_M.gguf`, is exactly this: a 1.7B-parameter Qwen3 quantized to
  4-bit GGUF running under llama.cpp with an 8k-token context window).
- **MoE (Mixture-of-Experts).** Some models are *sparsely activated*: instead of
  running all their weights for every token, an MoE model has many small "expert"
  subnetworks and a **router** that activates only a few of them per token. This
  gives a large effective parameter count while keeping per-token compute (and
  speed) low. For RAG, MoE models matter because they can afford a big context
  window and rich reasoning without being prohibitively slow on local hardware.

The practical upshot for this app: the **context window** of your chosen local model
dictates how big the retrieved context can be (`CONTEXT_WORD_BUDGET`). A small model
with a tight window (e.g. 1.7B / 8k tokens) needs a smaller budget; a large 32k+
model can handle more evidence.

### Agentic tool-calling loop (optional)

For the non-streaming `/api/chat` path, the LLM can go one step further and *act as
an agent* (`agentic_enabled`). It's given access to tools it can call mid-conversation:

- **`search_library`** — run another retrieval search
- **`get_section`** — pull a full parent section by ID
- **`summarize_work`** — summarize a specific named book

The loop runs up to `agentic_max_steps: 3` turns: the model either calls a tool (the
result is fed back to it) or produces a final answer. If the model's server doesn't
support function-calling, it falls back to a **ReAct** text protocol, where the model
emits `call: search_library("...")` lines that the program parses and executes.

---

## Part 7 — Other ways to use the library

- **MCP server** (`run_mcp.sh`): exposes the index as *callable tools*
  (`search_library`, `summarize_work`, `list_collections`) so an MCP-compatible chat
  client like LM Studio can ground its answers in your library. The client's own
  model does the chatting and calls the tools as needed.
- **Agent CLI** (`scripts/agent.py`): chat from the terminal.

---

## Part 8 — How the retrieval settings were tuned

The many knobs above weren't picked at random — they were tuned against a **golden
set** of known-good question→answer pairs in `evals/golden.jsonl`. The included
eval harness (`scripts/eval.py`) measures retrieval quality with no LLM calls,
so it's fast and deterministic:

- **hit@K** — did the right source surface in the top K results?
- **mrr@K** (mean reciprocal rank) — how high did it rank?
- **context_recall** — did the right source make it into the final LLM context?
- **context_precision** — of the sources shown, what fraction were relevant?

The workflow is: tweak a setting → `eval.py run` → compare the numbers → keep the
change only if it helps. `eval.py diff` acts as a **CI regression gate**: it fails
if any metric drops more than a tolerance vs. the saved baseline, so future changes
can't silently make retrieval worse. `tuning.md` documents which settings are
safe to change and which depend on your dataset or hardware. This is how the
defaults (chunk sizes, RRF k, relevance threshold, context budget) were validated
as sensible rather than guessed.

---

## Summary: the full chain in one picture

```
Raw book (PDF/EPUB) 
   │  extract text (OCR if scanned)
   ▼
Text 
   │  chunk into ~330-token pieces, overlap 60 (parent–child linking)
   ▼
Chunks 
   │  embed with E5 "passage:" prefix → 384-dim vectors
   ▼
ChromaDB vector index + manifest.db (metadata, BM25 index, parents)
   │
Your question 
   │  embed with E5 "query:" prefix
   ▼
Dense nearest-neighbor search     +    BM25 lexical search
   │                                        │
   └───────── Reciprocal Rank Fusion ───────┘
                     │  cross-encoder rerank
                     │  low-relevance guard
                     │  diversify + trim to context budget
                     ▼
Top chunks → SYSTEM_PROMPT + history + user question → local LLM → streamed answer
                     │                                    │
                     └── sources, scores, fiction warning ─┘
```

Every stage has a job: **chunking** keeps meaning intact, **embeddings** enable
semantic and multilingual search, **BM25** catches exact keywords, **fusion and
reranking** combine and sharpen the candidates, **multi-hop** broadens the search,
and the **LLM** turns the best evidence into a readable, honestly-sourced answer.
```