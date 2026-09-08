# RAG Library Agent

A local, privacy-first document assistant with a web GUI. Index your PDFs / EPUBs /
MOBI library and ask questions about it — everything runs on **your machine**, no cloud.

It is *fiction-aware*: if a book is tagged `Fiction` / `Short Stories` in Calibre, the
assistant treats it as fiction and will not present its content as fact.

```
┌─────────────┐   embeddings    ┌──────────────┐   retrieved   ┌────────────┐
│ PDF/EPUB/MOBI│ ─────────────► │ Chroma vector │ ────────────► │  LLM 1     │
│  library    │   (local)       │   store       │    context    │ (LM Studio)│
└─────────────┘                 └──────────────┘               └────────────┘
```

## Quick start (new machine)

Prereq: **Python 3.10+** on Linux, and **LM Studio** (or any OpenAI-compatible server)
if you want to chat.

```bash
cd rag-web-chat
./run.sh
```

That one command creates a virtualenv, installs dependencies, downloads the embedding
model once, and starts the web app at **http://localhost:5000**.

If `./run.sh` isn't executable yet: `chmod +x run.sh`

### Manual / troubleshooting
```bash
# set up env explicitly
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# start the web app
.venv/bin/python scripts/server.py
# change port:   RAG_PORT=8080 .venv/bin/python scripts/server.py
```

## Using the app

The app has 4 tabs:

1. **Setup** — pick your library folder (use the **Browse…** folder picker, or type a
   path), set the embedding model, chunk sizes, LLM URL, OCR toggle. `Save config`
   persists to `config.json`.
2. **Scan** — dry-run: counts files by type, classifies Fiction vs Non-Fiction from
   Calibre tags, and flags scanned PDFs that would need OCR. Nothing is indexed.
3. **Ingest** — builds/updates the vector index. Runs in the background; a header pill
   and the Ingest tab show live progress on any tab. You can **Pause / Resume / Stop** a
   running job. Re-run anytime to pick up new files (incremental). While indexing runs,
   **Chat is paused** (to avoid ChromaDB concurrent read/write errors) and re-enables
   when the job finishes or is stopped.
4. **Chat** — ask questions. Sources are shown with kind (fiction/non-fiction) tags and
   similarity scores. If only fiction sources were found, the app visibly warns you.

### What does the index store / where?
| Thing             | Location              |
|-------------------|-----------------------|
| config            | `config.json`         |
| vector index      | `index/`              |
| file status       | `manifest.db`         |
| scan JSON         | `scan_results.json`   |
| ingest log        | `ingest.log`          |

Delete the `index/` + `manifest.db` pair to rebuild from scratch (e.g. after changing
the embedding model; the GUI's `Force re-embed` does this per file too).

## Config reference (`config.json`)

| Key                  | Default                            | Meaning |
|----------------------|------------------------------------|---------|
| `embed_model`        | `intfloat/multilingual-e5-small`   | Sentence-Transformers model |
| `embed_device`       | `cpu`                              | `cpu` (GPU needs a big VRAM card) |
| `chunk_tokens`       | `330`                              | tokens per chunk (keep ≤ model max, e.g. 512 for e5) |
| `chunk_overlap`      | `60`                               | overlap between chunks |
| `llm_base_url`       | `http://localhost:1234/v1`         | LM Studio base URL |
| `llm_model`          | `default`                          | model id to request (the shipped `config.json` sets a concrete model) |
| `retrieval_top_k`    | `10`                               | chunks retrieved per question |
| `fiction_tags`       | `["Fiction","Short Stories","Literary"]` | Calibre tags that mark a book as fiction |
| `ocr_enabled`        | `false`                            | OCR scanned PDFs (needs extra deps) |
| `ocr_char_threshold` | `50`                               | min text chars before a PDF is "scanned" |

## Notes / limitations

- **Scanned PDFs**: without OCR enabled they are skipped and recorded (they show up in
  the Scan tab). Enabling OCR requires installing `rapidocr-onnxruntime` (already in
  `requirements.txt`) — it is faster but still much slower than text PDFs.
- **Very large books**: single files that chunk into more than ~5000 chunks (e.g.
  complete-works omnibuses) are skipped with a `SKIP (oversized…)` message, because
  Chroma upserts have a hard batch limit (~5461). Override with `INGEST_MAX_CHUNKS`.
- **Fiction classification**: relies on Calibre tags. Untagged books default to
  non-fiction; edit tags in Calibre (*Fiction*) and re-scan.
- **Embedding model choice**: `multilingual-e5-small` is a good CPU default (~10
  files/min). Bigger models (`bge-m3`, `Qwen3`) give better quality but are 4–10×
  slower on CPU. See `scripts/bench_fast.py`.
- **Other people's setup**: this repo/ folder is fully portable — copy the whole RAG
  folder to the new machine and run `./run.sh`. The library itself stays wherever it is.

## CLI (power users)

```bash
.venv/bin/python scripts/scan.py  /path/to/library             # dry-run report
.venv/bin/python scripts/ingest.py /path/to/library --set library   # index
.venv/bin/python scripts/agent.py --set library             # chat in terminal
```