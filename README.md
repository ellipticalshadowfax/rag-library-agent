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
if you want to chat. **uv** is auto-installed on first run (or install manually:
`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
cd rag-web-chat
./run.sh                    # CPU install (default, no CUDA libs needed)
RAG_DEVICE=gpu ./run.sh     # GPU install (NVIDIA driver + VRAM required)
```

That one command creates a virtualenv, installs dependencies, downloads the embedding
model once, and starts the web app at **http://localhost:5000**.

If `./run.sh` isn't executable yet: `chmod +x run.sh`

### Manual / troubleshooting
```bash
# set up env explicitly (CPU install — two steps required: torch CPU first)
python3 -m venv .venv
uv pip install torch==2.14.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt \
  --extra-index-url https://pypi.org/simple

# GPU install (NVIDIA driver + VRAM required)
RAG_DEVICE=gpu ./run.sh
# or manually: uv pip install -r requirements-gpu.txt

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

- **CPU vs GPU install**: `RAG_DEVICE=gpu` installs CUDA-capable wheels (`torch` with
  CUDA, `nvidia-*` libraries). **However, the app currently forces CPU at runtime**
  (`CUDA_VISIBLE_DEVICES=""` is hardcoded), so embeddings run on CPU regardless of which
  install was used. The GPU install provisions the libraries for future use only — no
  GPU speedup today.
- **OCR is CPU-only** in both install modes — it uses `onnxruntime` (CPU), not
  `onnxruntime-gpu`. GPU acceleration for OCR is not currently supported.
- **Scanned PDFs**: without OCR enabled they are skipped and recorded (they show up in
  the Scan tab). Enabling OCR requires installing `rapidocr-onnxruntime` (already in
  `requirements.txt`) — it is faster but still much slower than text PDFs.

## OCR engine comparison (`scripts/ocr_compare.py`)

`ingest.py` handles a scanned PDF by running one OCR engine over every page. To decide
which engine is best for a given library, `ocr_compare.py` samples the first few pages
of each OCR-needing PDF, scores output quality with both **Tesseract** and **RapidOCR**,
routes each file to the better engine, then full-OCRs it into `ocr/<stem>.txt` (the cache
`ingest.py` already reads).

```bash
# compare only (no full OCR); write report to ocr_compare_report.json
.venv/bin/python scripts/ocr_compare.py /path/to/library --sample-only

# compare + full-OCR every file with its winning engine
.venv/bin/python scripts/ocr_compare.py /path/to/library

# only process files matching a substring (e.g. one author / one book)
.venv/bin/python scripts/ocr_compare.py /path/to/library --only "Ansel Adams"
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--sample N` | `5` | pages sampled per file for quality scoring |
| `--workers N` | `8` | parallel OCR page workers (full pass). Tesseract uses ~1 core/worker; RapidOCR roughly half. |
| `--sample-only` | off | run the comparison only, skip full-file OCR |
| `--force` | off | re-OCR files that already have a cache |
| `--tess-threshold` / `--rapid-threshold` | `0.50` / `0.45` | min sample quality to accept an engine |
| `--tesseract` / `--tessdata` | auto | paths to the tesseract binary / tessdata |

**Threading**: the full OCR pass is multi-threaded across pages — bump `--workers` to use
more cores (e.g. `--workers 16` on a 16-core box). Tesseract spawns one subprocess per
worker, so it scales linearly with `--workers`; RapidOCR is CPU-heavy and shares cores.
The 5-page *sampling* pass is single-threaded by design. OCR results are cached per file,
so a run interrupted mid-way resumes on re-run without `--force`.
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
.venv/bin/python scripts/ocr_compare.py /path/to/library       # pick OCR engine + OCR scans
```