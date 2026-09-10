#!/usr/bin/env python3
"""ocr.py - standalone OCR for a directory of scanned PDFs.

Writes the OCR text back into the originals:
  --mode merge   : add an invisible text layer to each PDF (default, "auto write back")
  --mode sidecar : write <name>.txt next to each PDF, leaving the PDF untouched

Usage:
  .venv/bin/python scripts/ocr.py /path/to/pdfs [--mode merge|sidecar] [--force] [--only SUBSTR] [--languages en,...]
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import ingest as ING  # needs_ocr, ocr_pdf_rapidocr, merge_text_into_pdf

from _paths import rag_root

RAG_ROOT = rag_root()
OCR_CACHE = RAG_ROOT / "ocr"
OCR_LOCK = RAG_ROOT / ".ocr.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ocr_running() -> bool:
    try:
        d = json.loads(OCR_LOCK.read_text())
        return _pid_alive(int(d.get("pid", -1)))
    except Exception:
        return False


def acquire_lock(target: str, mode: str) -> bool:
    if ocr_running():
        return False
    OCR_LOCK.write_text(json.dumps({
        "pid": os.getpid(), "target": str(target), "mode": mode,
        "started_at": time.time(),
    }))
    return True


def release_lock():
    try:
        OCR_LOCK.unlink()
    except OSError:
        pass


def _handle_stop(signum, frame):
    print("\nOCR stopped by user. Finalizing...", flush=True)
    raise SystemExit(0)


def run(target: str, mode: str, cfg: dict, force: bool = False,
        only_path: str = None, languages=None, backend=None):
    target_path = Path(target)
    if not target_path.is_dir():
        print(f"Error: {target} is not a directory", flush=True)
        sys.exit(1)

    threshold = cfg.get("ocr_char_threshold", 50)
    langs = languages or cfg.get("ocr_languages", ["en"])
    backend = backend or cfg.get("ocr_backend", "tesseract")
    OCR_CACHE.mkdir(exist_ok=True)

    # Collect PDFs that need OCR (skip hidden and underscore-prefixed dirs).
    files = []
    for root, dirs, fnames in os.walk(target_path):
        dirs[:] = [d for d in dirs if not (d.startswith(".") or d.startswith("_"))]
        for fn in fnames:
            p = Path(root) / fn
            if p.suffix.lower() != ".pdf":
                continue
            rel = str(p.relative_to(target_path))
            if only_path and only_path not in rel:
                continue
            if not force and not ING.needs_ocr(str(p), threshold):
                continue
            files.append(p)

    total = len(files)
    print(f"[bold]{total} PDFs to OCR (mode={mode})[/bold]".replace("[bold]", "").replace("[/bold]", ""), flush=True)

    done = skipped = errors = 0
    start = time.time()
    for idx, p in enumerate(files, start=1):
        rel = str(p.relative_to(target_path))
        try:
            if mode == "sidecar":
                side = p.with_suffix(".txt")
                if side.exists() and not force:
                    print(f"[{idx}/{total}] SKIP (has sidecar): {p.name}", flush=True)
                    skipped += 1
                    continue
                ok = ING.ocr_pdf_auto(str(p), str(side), langs, backend)
                print(f"[{idx}/{total}] {'sidecar' if ok else 'failed'}: {p.name}", flush=True)
            else:  # merge back into the original PDF
                cache = OCR_CACHE / (p.stem + ".txt")
                ok = ING.ocr_pdf_auto(str(p), str(cache), langs, backend)
                if ok:
                    merged = ING.merge_text_into_pdf(str(p), str(cache))
                    print(f"[{idx}/{total}] {'merged' if merged else 'merge-failed'}: {p.name}", flush=True)
                    ok = merged
            if ok:
                done += 1
            else:
                errors += 1
        except Exception as e:
            print(f"[{idx}/{total}] ERROR {rel}: {e}", flush=True)
            errors += 1

        elapsed = time.time() - start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = f"eta={int((total-idx)//rate//60)}m{int((total-idx)//rate%60)}s" if rate > 0 else "eta=?"
        print(f"[{idx}/{total}] done={done} skip={skipped} err={errors} "
              f"elapsed={int(elapsed//60)}m{int(elapsed%60)}s {eta}", flush=True)

    print(f"\nOCR complete: done={done} skip={skipped} errors={errors}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OCR scanned PDFs in a directory")
    parser.add_argument("target", nargs="?", help="Directory to OCR")
    parser.add_argument("--mode", choices=["merge", "sidecar"], default="merge",
                        help="merge=write text layer into the PDF (default); sidecar=write .txt only")
    parser.add_argument("--force", action="store_true", help="OCR even if a text layer already exists")
    parser.add_argument("--only", help="Only process paths matching this substring")
    parser.add_argument("--languages", help="Comma-separated OCR languages (default from config)")
    parser.add_argument("--backend", choices=["tesseract", "rapidocr"], default=None,
                        help="OCR engine (default from config ocr_backend)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    cfg = ING.load_config()
    target = args.target
    if not target:
        parser.error("a target directory is required")

    if not acquire_lock(target, args.mode):
        print(f"Another OCR is already running (lock held).", flush=True)
        sys.exit(1)

    try:
        run(target, args.mode, cfg, force=args.force, only_path=args.only,
            languages=(args.languages.split(",") if args.languages else None),
            backend=args.backend)
    finally:
        release_lock()


if __name__ == "__main__":
    main()