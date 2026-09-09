#!/usr/bin/env python3
"""ingest.py - Incremental RAG indexing with fiction-aware metadata."""

import hashlib
import json
import os
import random
import re
import signal
import sqlite3
import subprocess
import sys
import time
import random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for embeddings

import chromadb
import fitz  # PyMuPDF
import torch
from rich.console import Console

console = Console()

EXTENSIONS_TEXT = {".pdf", ".epub", ".mobi", ".djvu", ".txt", ".html", ".htm"}

INGEST_LOCK = Path(__file__).resolve().parent.parent / ".ingest.lock"

# Chroma upserts have a hard batch limit (~5461 embeddings per call). Instead of
# failing the whole embed pass, we always upsert in batches of INGEST_BATCH_SIZE.
# Files that chunk into more than MAX_CHUNKS_PER_FILE are flagged as "big" and
# processed across multiple batches rather than skipped. Overrides:
#   INGEST_BATCH_SIZE (per-call batch, default 500)
#   INGEST_MAX_CHUNKS  (big-file notice threshold, default 5000)
MAX_CHUNKS_PER_FILE = int(os.environ.get("INGEST_MAX_CHUNKS", "5000"))
INGEST_BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "500"))


# ─── Ingest lock (cross-process: blocks chat while indexing) ─────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_ingest_lock() -> dict | None:
    """Return lock info if an ingest is currently running (lock + live PID)."""
    try:
        data = json.loads(INGEST_LOCK.read_text())
        if _pid_alive(int(data.get("pid", -1))):
            return data
    except Exception:
        pass
    return None


def ingest_running() -> bool:
    return read_ingest_lock() is not None


def acquire_ingest_lock(set_name: str, target: str) -> bool:
    if ingest_running():
        return False
    INGEST_LOCK.write_text(json.dumps({
        "pid": os.getpid(),
        "started_at": time.time(),
        "set": set_name,
        "target": str(target),
    }))
    return True


def release_ingest_lock():
    try:
        INGEST_LOCK.unlink()
    except OSError:
        pass


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(cfg_path) as f:
        return json.load(f)

# ─── Manifest DB ─────────────────────────────────────────────────────────────

class Manifest:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rel_path TEXT NOT NULL UNIQUE,
                abs_path TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                ocr_status TEXT DEFAULT 'none',
                kind TEXT DEFAULT 'unknown',
                tags TEXT DEFAULT '[]',
                title TEXT DEFAULT '',
                set_name TEXT DEFAULT '',
                indexed_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_files_path ON files(rel_path);
        """)
        self.conn.commit()

    def is_current(self, rel_path: str, mtime: float, size: int) -> bool:
        row = self.conn.execute(
            "SELECT mtime, size FROM files WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return row and row["mtime"] == mtime and row["size"] == size

    def upsert(self, rel_path, abs_path, mtime, size, ocr_status="none",
               kind="unknown", tags=None, title="", set_name=""):
        tags_json = json.dumps(tags or [])
        self.conn.execute("""
            INSERT INTO files (rel_path, abs_path, mtime, size, ocr_status, kind, tags, title, set_name, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rel_path) DO UPDATE SET
                abs_path=excluded.abs_path,
                mtime=excluded.mtime,
                size=excluded.size,
                ocr_status=excluded.ocr_status,
                kind=excluded.kind,
                tags=excluded.tags,
                title=excluded.title,
                set_name=excluded.set_name,
                indexed_at=excluded.indexed_at
        """, (rel_path, abs_path, mtime, size, ocr_status, kind, tags_json, title, set_name, time.time()))
        self.conn.commit()

    def remove(self, rel_path: str):
        self.conn.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))
        self.conn.commit()

    def get_indexed_paths(self) -> set[str]:
        rows = self.conn.execute("SELECT rel_path FROM files").fetchall()
        return {r["rel_path"] for r in rows}

    def close(self):
        self.conn.close()

# ─── Calibre Metadata ────────────────────────────────────────────────────────

def calibre_db_path(target: Path) -> Path | None:
    # Check parents first (standard Calibre layout)
    p = target
    while p != p.parent:
        db = p / "metadata.db"
        if db.exists():
            return db
        p = p.parent
    # Also search within target (metadata.db may be in a subdirectory)
    for db in target.rglob("metadata.db"):
        return db
    return None


def load_calibre_tags(db_path: Path) -> dict[str, dict]:
    """Return {filename: {title, tags}} from Calibre metadata.db.

    Keys include both the stem (no ext) and the full filename so lookups
    work regardless of whether the caller passes "foo.epub" or "foo".
    """
    meta = {}
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("""
            SELECT d.name, d.format,
                   (SELECT title FROM books WHERE id=d.book) as title,
                   GROUP_CONCAT(t.name, '|||') as tags
            FROM data d
            LEFT JOIN books_tags_link btl ON btl.book = d.book
            LEFT JOIN tags t ON btl.tag = t.id
            GROUP BY d.id
        """)
        for row in cur.fetchall():
            dname, fmt, title, tag_str = row
            tags = [t.strip() for t in tag_str.split("|||") if t.strip()] if tag_str else []
            info = {
                "title": title or dname,
                "tags": tags,
                "format": fmt,
            }
            # Index by stem (no ext) so "The Call of Cthulhu - H. P. Lovecraft" matches
            meta[dname] = info
            # Also index by common filename variants with extension
            if fmt:
                meta[f"{dname}.{fmt.lower()}"] = info
                meta[f"{dname}.{fmt.upper()}"] = info
        conn.close()
    except Exception as e:
        console.print(f"[yellow]Warning: Calibre DB read failed: {e}[/yellow]")
    return meta


def load_opf_metadata(root: Path) -> dict[str, dict]:
    """Fallback: read metadata.opf files."""
    meta = {}
    for opf in root.rglob("metadata.opf"):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(opf)
            ns = {"dc": "http://purl.org/dc/elements/1.1/"}
            title_el = tree.find(".//dc:title", ns)
            title = title_el.text if title_el is not None else opf.parent.name
            subjects = tree.findall(".//dc:subject", ns)
            tags = [s.text for s in subjects if s.text]
            parent = opf.parent
            for f in parent.iterdir():
                if f.is_file() and f.suffix.lower() in EXTENSIONS_TEXT:
                    meta[f.name] = {"title": title, "tags": tags, "rel_dir": str(parent.relative_to(root))}
                    break
        except Exception:
            pass
    return meta


def classify_kind(tags: list[str], fiction_tags: list[str]) -> str:
    tag_set = {t.lower() for t in tags}
    fiction_set = {t.lower() for t in fiction_tags}
    if tag_set & fiction_set:
        return "fiction"
    return "nonfiction"

# ─── OCR ─────────────────────────────────────────────────────────────────────

def needs_ocr(pdf_path: str, threshold: int = 50) -> bool:
    try:
        result = subprocess.run(
            ["pdftotext", "-l", "3", pdf_path, "-"],
            capture_output=True, text=True, timeout=15
        )
        return len(result.stdout.strip()) < threshold
    except Exception:
        return True


def ocr_pdf_rapidocr(pdf_path: str, output_path: str, languages=["en"]):
    """OCR a scanned PDF using pdftoppm + rapidocr-onnxruntime -> text file."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        console.print("[yellow]rapidocr not installed; skipping OCR. Install with: pip install rapidocr-onnxruntime[/yellow]")
        return False

    ocr = RapidOCR()
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image
        pix = page.get_pixmap(dpi=200)
        img_data = pix.tobytes("png")

        # Write to temp file for rapidocr
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name

        try:
            result, _ = ocr(tmp_path)
            page_text = "\n".join([line[1] for line in result]) if result else ""
            all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
        finally:
            os.unlink(tmp_path)

    doc.close()

    # Save OCR text
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    return True


def ocr_pdf_auto(pdf_path: str, output_path: str, languages=["en"], backend=None):
    """Dispatch OCR to the configured backend (tesseract default, or rapidocr)."""
    if backend == "rapidocr":
        return ocr_pdf_rapidocr(pdf_path, output_path, languages)
    return ocr_pdf_tesseract(pdf_path, output_path, languages)


def ocr_pdf_tesseract(pdf_path: str, output_path: str, languages=["en"]) -> bool:
    """OCR a scanned PDF with tesseract (via ocr_compare's parallel helper)."""
    try:
        import ocr_compare as OC
    except Exception as e:
        console.print(f"[yellow]ocr_compare import failed: {e}[/yellow]")
        return False
    if not OC._setup_tesseract(None, None):
        console.print("[yellow]tesseract binary not available; install tesseract-ocr or set TESSERACT_BIN[/yellow]")
        return False
    lang = OC._tess_lang(languages)
    ok = OC.ocr_full_tesseract(pdf_path, lang, Path(output_path))
    console.print(f"[dim]  tesseract OCR {'ok' if ok else 'no text'}: {Path(pdf_path).name}[/dim]")
    return ok


def merge_text_into_pdf(pdf_path: str, text_path: str) -> bool:
    """Write the OCR text from text_path back into pdf_path as an invisible text
    layer so the PDF becomes searchable/extractable without changing its look.

    Page markers "--- Page N ---" (written by ocr_pdf_rapidocr) attach each page's
    text to the corresponding page. A backup of the original is kept next to the
    OCR cache. Returns True on success.
    """
    def parse_pages(text: str) -> dict:
        pages = {}
        cur = None
        buf = []
        pat = re.compile(r"^--- Page (\d+) ---\s*$")
        for line in text.splitlines():
            m = pat.match(line)
            if m:
                if cur is not None:
                    pages[cur] = "\n".join(buf).strip()
                cur = int(m.group(1))
                buf = []
            else:
                buf.append(line)
        if cur is not None:
            pages[cur] = "\n".join(buf).strip()
        return pages

    try:
        text = open(text_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False
    page_texts = parse_pages(text)
    if not page_texts:
        return False

    try:
        doc = fitz.open(pdf_path)
        added = 0
        for i in range(len(doc)):
            txt = page_texts.get(i + 1, "")
            if not txt.strip():
                continue
            page = doc[i]
            r = page.rect + (-6, -6, 6, 6)
            # render_mode 3 = invisible glyphs: searchable/copyable, not drawn.
            page.insert_textbox(r, txt, fontsize=9.0, fontname="helv",
                                render_mode=3, align=0)
            added += 1
        doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()
        return added > 0
    except Exception as e:
        console.print(f"[red]merge_text_into_pdf failed for {pdf_path}: {e}[/red]")
        return False

# ─── Text Extraction ─────────────────────────────────────────────────────────

def extract_text(fpath: Path, ext: str, ocr_cache: Path) -> tuple[str, dict]:
    """Extract text from a file. Returns (text, {page_texts})."""
    meta = {"pages": []}

    if ext == ".pdf":
        # Check for OCR cache first
        ocr_cached = ocr_cache / (fpath.stem + ".txt")
        if ocr_cached.exists():
            text = ocr_cached.read_text(encoding="utf-8", errors="replace")
            return text, meta

        try:
            doc = fitz.open(str(fpath))
            texts = []
            for page in doc:
                texts.append(page.get_text())
            doc.close()
            text = "\n\n".join(texts)
            if len(text.strip()) > 50:
                return text, meta
        except Exception:
            pass

    elif ext == ".epub":
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.result = []
                    self._skip = 0
                def handle_starttag(self, tag, attrs):
                    if tag in ("style", "script"):
                        self._skip += 1
                def handle_endtag(self, tag):
                    if tag in ("style", "script"):
                        self._skip = max(0, self._skip - 1)
                def handle_data(self, data):
                    if self._skip == 0:
                        self.result.append(data)
                def get_text(self):
                    return " ".join(self.result)

            book = epub.read_epub(str(fpath))
            texts = []
            # Some EPUBs tag their content as ITEM_UNKNOWN instead of ITEM_DOCUMENT,
            # so match on filename rather than the (unreliable) type label.
            docs = [i for i in book.get_items()
                    if i.get_name().lower().endswith((".html", ".xhtml", ".htm"))]
            for item in docs:
                ext_parser = TextExtractor()
                ext_parser.feed(item.get_content().decode("utf-8", errors="replace"))
                t = ext_parser.get_text().strip()
                if t:
                    texts.append(t)
            text = "\n\n".join(texts)
            return text, meta
        except Exception as e:
            console.print(f"[yellow]EPUB extraction failed for {fpath}: {e}[/yellow]")

    elif ext == ".txt" or ext == ".html" or ext == ".htm":
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
            return text, meta
        except Exception:
            pass

    elif ext == ".djvu":
        # Try pdftotext (won't work) or djvutxt if available
        try:
            result = subprocess.run(["djvutxt", str(fpath)], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout, meta
        except Exception:
            pass
        # Fallback: PyMuPDF can't read DJVU, note it needs external tool
        console.print(f"[yellow]DJVU extraction needs djvutxt: {fpath}[/yellow]")

    elif ext == ".mobi":
        try:
            import mobi
            tempdir, filepath = mobi.extract(str(fpath))
            from pathlib import Path as P
            html_files = list(P(tempdir).rglob("*.html")) + list(P(tempdir).rglob("*.htm"))
            texts = []
            for hf in html_files:
                texts.append(hf.read_text(encoding="utf-8", errors="replace"))
            import shutil
            shutil.rmtree(tempdir, ignore_errors=True)
            text = "\n\n".join(texts)
            return text, meta
        except ImportError:
            console.print(f"[yellow]mobi package not installed; skipping {fpath}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]MOBI extraction failed for {fpath}: {e}[/yellow]")

    return "", meta

# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_tokens: int = 500, overlap: int = 100) -> list[str]:
    """Split text into chunks by word count (rough token approximation)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_tokens
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start >= len(words):
            break
    return chunks

# ─── Main Ingestion ──────────────────────────────────────────────────────────

def ingest(target: str, set_name: str, cfg: dict, force: bool = False, only_path: str = None):
    target_path = Path(target)
    if not target_path.is_dir():
        console.print(f"[red]Error: {target} is not a directory[/red]")
        sys.exit(1)

    rag_root = Path(__file__).resolve().parent.parent
    ocr_cache = rag_root / "ocr"
    cache_dir = rag_root / "cache"
    index_dir = rag_root / "index"

    ocr_cache.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    fiction_tags = cfg.get("fiction_tags", ["Fiction", "Short Stories", "Literary"])
    chunk_tokens = cfg.get("chunk_tokens", 500)
    chunk_overlap = cfg.get("chunk_overlap", 100)
    embed_model = cfg.get("embed_model", "BAAI/bge-m3")
    ocr_threshold = cfg.get("ocr_char_threshold", 50)
    ocr_enabled = cfg.get("ocr_enabled", False)
    ocr_merge = cfg.get("ocr_merge", True)
    batch_size = int(cfg.get("ingest_batch_size", INGEST_BATCH_SIZE))
    exclude = cfg.get("exclude", [])

    if ocr_enabled:
        console.print("[dim]OCR: enabled[/dim]")
    else:
        console.print("[dim]OCR: disabled (files without text layer will be skipped)[/dim]")

    # Load Calibre metadata
    cal_db = calibre_db_path(target_path)
    cal_tags = {}
    if cal_db:
        console.print(f"[dim]Calibre DB: {cal_db}[/dim]")
        cal_tags = load_calibre_tags(cal_db)

    # Load manifest
    manifest = Manifest(rag_root / "manifest.db")

    # Load ChromaDB
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(
        name=set_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Initialize embedder
    console.print(f"[dim]Loading embedding model: {embed_model}...[/dim]")
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(os.cpu_count() or 8)
    embedder = SentenceTransformer(embed_model, device=cfg.get("embed_device", "cpu"))
    if hasattr(embedder, "prompts") and "passage" not in embedder.prompts:
        embedder.prompts.update({"passage": "passage: ", "query": "query: "})
    console.print(f"[dim]Embedder ready (dim={embedder.get_sentence_embedding_dimension()}, "
                  f"threads={torch.get_num_threads()})[/dim]")

    # Collect files
    files_to_process = []
    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext not in EXTENSIONS_TEXT:
                continue

            rel = str(fpath.relative_to(target_path))
            if only_path and only_path not in rel:
                continue
            if exclude and any(x in rel for x in exclude):
                continue

            stat = fpath.stat()
            mtime = stat.st_mtime
            size = stat.st_size

            if not force and manifest.is_current(rel, mtime, size):
                continue

            # Get metadata
            cal_meta = cal_tags.get(fname, {})
            title = cal_meta.get("title", fpath.stem)
            tags = cal_meta.get("tags", [])
            kind = classify_kind(tags, fiction_tags)

            # Check OCR
            ocr_status = "none"
            if (
                ocr_enabled
                and ext == ".pdf"
                and needs_ocr(str(fpath), ocr_threshold)
            ): 
                ocr_status = "needed"
            elif ext == ".pdf" and needs_ocr(str(fpath), ocr_threshold):
                ocr_status = "needs_ocr_skipped"

            if ocr_status == "needs_ocr_skipped":
                # OCR disabled and no text layer — mark as known-skipped
                manifest.upsert(rel, str(fpath), mtime, size,
                               "needs_ocr_skipped", kind, tags, title, set_name)
                continue

            files_to_process.append({
                "path": rel,
                "full_path": str(fpath),
                "ext": ext,
                "mtime": mtime,
                "size": size,
                "title": title,
                "tags": tags,
                "kind": kind,
                "ocr_status": ocr_status,
            })

    if not files_to_process:
        console.print("[green]No new or changed files to process.[/green]")
        manifest.close()
        return

    # Process in random order so no single file's slow embed blocks progress
    random.shuffle(files_to_process)

    console.print(f"\n[bold]{len(files_to_process)} files to process (random order)[/bold]")
    console.print(f"  Fiction: {sum(1 for f in files_to_process if f['kind'] == 'fiction')}")
    console.print(f"  Non-fiction: {sum(1 for f in files_to_process if f['kind'] == 'nonfiction')}")
    console.print(f"  Need OCR: {sum(1 for f in files_to_process if f['ocr_status'] == 'needed')}")
    console.print()

    processed = 0
    skipped = 0
    errors = 0
    total = len(files_to_process)
    start_time = time.time()

    for idx, finfo in enumerate(files_to_process, start=1):
        rel = finfo["path"]
        fpath = Path(finfo["full_path"])
        ext = finfo["ext"]

        try:
            # OCR if needed
            if finfo["ocr_status"] == "needed":
                ocr_out = ocr_cache / (fpath.stem + ".txt")
                if not ocr_out.exists():
                    print(f"[{idx}/{total}] OCR: {fpath.name}", flush=True)
                    ocr_pdf_auto(str(fpath), str(ocr_out), cfg.get("ocr_languages", ["en"]),
                                 cfg.get("ocr_backend", "tesseract"))
                    finfo["ocr_status"] = "done"
                else:
                    finfo["ocr_status"] = "cached"
                # Auto write-back: merge the OCR text into the original PDF so it
                # becomes searchable/extractable without an OCR re-run next time.
                if ocr_merge:
                    merged = merge_text_into_pdf(str(fpath), str(ocr_out))
                    print(f"[{idx}/{total}] {'merged' if merged else 'merge-failed'} OCR into {fpath.name}", flush=True)

            # Extract text
            text, text_meta = extract_text(fpath, ext, ocr_cache)
            if not text.strip():
                print(f"[{idx}/{total}] SKIP (no text): {fpath.name}", flush=True)
                manifest.upsert(rel, str(fpath), finfo["mtime"], finfo["size"],
                               finfo["ocr_status"], finfo["kind"], finfo["tags"], finfo["title"], set_name)
                skipped += 1
                continue

            # Early, cheap estimate (chars / ~4 chars per token / chunk_tokens) so a
            # huge file is flagged before the chunk/embed pass spends time on it.
            est_chunks = max(1, len(text) // max(1, chunk_tokens * 4))
            if est_chunks > MAX_CHUNKS_PER_FILE:
                console.print(f"[yellow][{idx}/{total}] Big file detected: {fpath.name} "
                              f"(~{est_chunks} chunks) - will process in batches[/yellow]")

            # Chunk
            chunks = chunk_text(text, chunk_tokens, chunk_overlap)
            if not chunks:
                print(f"[{idx}/{total}] SKIP (no chunks): {fpath.name}", flush=True)
                skipped += 1
                continue

            # Oversized single file (e.g. complete-works omnibus): handled below by
            # processing across multiple embed/upsert batches instead of skipping.
            if len(chunks) > MAX_CHUNKS_PER_FILE:
                print(f"[{idx}/{total}] BIG: {fpath.name} ({len(chunks)} chunks) - "
                      f"processing in batches of {batch_size}", flush=True)

            # Embed + upsert in batches. Chroma has a hard per-call limit (~5461), so
            # any file - including huge omnibuses - is processed across multiple calls
            # instead of a single one. Chunk IDs stay deterministic (rel:index).
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                with torch.no_grad():
                    embeddings = embedder.encode(batch, show_progress_bar=False,
                                                 convert_to_numpy=True,
                                                 prompt_name="passage").tolist()

                all_ids = []
                all_docs = []
                all_embeddings = []
                all_metadatas = []
                for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                    abs_index = start + j
                    chunk_id = hashlib.sha256(f"{rel}:{abs_index}".encode()).hexdigest()[:16]
                    all_ids.append(chunk_id)
                    all_docs.append(chunk)
                    all_embeddings.append(emb)
                    all_metadatas.append({
                        "source": rel,
                        "title": finfo["title"],
                        "kind": finfo["kind"],
                        "tags": json.dumps(finfo["tags"]),
                        "set": set_name,
                        "chunk_index": abs_index,
                        "total_chunks": len(chunks),
                    })

                collection.upsert(
                    ids=all_ids,
                    documents=all_docs,
                    embeddings=all_embeddings,
                    metadatas=all_metadatas,
                )

            # Update manifest
            manifest.upsert(rel, str(fpath), finfo["mtime"], finfo["size"],
                           finfo["ocr_status"], finfo["kind"], finfo["tags"], finfo["title"], set_name)
            processed += 1

        except Exception as e:
            print(f"[{idx}/{total}] ERROR {rel}: {e}", flush=True)
            errors += 1

        # Status line every file
        elapsed = time.time() - start_time
        rate = idx / elapsed if elapsed > 0 else 0
        if rate > 0:
            eta = (total - idx) / rate
            eta_s = f"eta={int(eta//60)}m{int(eta%60)}s"
        else:
            eta_s = "eta=?" 
        print(f"[{idx}/{total}] done={processed} skip={skipped} err={errors} " 
              f"elapsed={int(elapsed//60)}m{int(elapsed%60)}s {eta_s}", flush=True)

    manifest.close()

    # Summary
    console.print()
    console.print("[bold]Ingestion complete[/bold]")
    console.print(f"  Processed: {processed}")
    console.print(f"  Skipped (no text): {skipped}")
    console.print(f"  Errors: {errors}")
    console.print(f"  Collection '{set_name}': {collection.count()} chunks")


def _handle_stop(signum, frame):
    console.print("\n[yellow]Ingest stopped by user. Finalizing...[/yellow]")
    raise SystemExit(0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents into RAG index")
    parser.add_argument("target", nargs="?", help="Directory to scan")
    parser.add_argument("--set", default="library", help="Collection/set name")
    parser.add_argument("--force", action="store_true", help="Reprocess all files")
    parser.add_argument("--only", help="Only process files matching this path substring")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    cfg = load_config()
    target = args.target or cfg["sets"][args.set]["path"]

    if not acquire_ingest_lock(args.set, str(target)):
        console.print("[red]Another ingest is already running (ingest.lock held). "
                      f"PID {read_ingest_lock().get('pid') if read_ingest_lock() else '?'}.[/red]")
        sys.exit(1)

    try:
        ingest(target, args.set, cfg, force=args.force, only_path=args.only)
    finally:
        release_ingest_lock()


if __name__ == "__main__":
    main()
