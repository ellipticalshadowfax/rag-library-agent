#!/usr/bin/env python3
"""merge_ocr_into_pdf.py - Add an invisible (searchable) text layer to scanned
PDFs using the OCR text already cached in ocr/<stem>.txt. Writes in place.

Safety: only touched when the OCR cache exists and page count matches.
"""

import re
import shutil
import sys
from pathlib import Path

import fitz  # PyMuPDF

RAG_ROOT = Path(__file__).resolve().parent.parent
OCR_CACHE = RAG_ROOT / "ocr"


def parse_ocr_text(text: str):
    """Split OCR cache text into {page_num: page_text} (1-based)."""
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


def add_text_layer(pdf_path: str, page_texts: dict):
    doc = fitz.open(pdf_path)
    n = len(doc)
    added = 0
    for i in range(n):
        txt = page_texts.get(i + 1, "")
        if not txt.strip():
            continue
        page = doc[i]
        rect = page.rect
        margin = 6
        r = fitz.Rect(rect.x0 + margin, rect.y0 + margin,
                      rect.x1 - margin, rect.y1 - margin)
        # Insert the OCR text as an invisible text layer (render mode 3 =
        # "neither fill nor stroke"). Glyphs are present so text is searchable
        # and copyable, but nothing is drawn on the page image. We use a small
        # font so the full page text fits within the textbox.
        page.insert_textbox(r, txt, fontsize=9.0, fontname="helv",
                            render_mode=3, align=0)
        added += 1
    doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    return n, added


def main():
    if len(sys.argv) < 2:
        print("usage: merge_ocr_into_pdf.py <ocr cache .txt ...>")
        sys.exit(1)
    for cache_arg in sys.argv[1:]:
        cache = Path(cache_arg)
        if not cache.exists():
            print(f"skip (no cache): {cache}")
            continue
        # find the source PDF by stem
        stem = cache.stem
        hits = []
        for root in ["/media/library"]:
            for f in Path(root).rglob("*.pdf"):
                if f.stem == stem:
                    hits.append(f)
        if not hits:
            print(f"skip (no source pdf): {cache.name}")
            continue
        pdf = hits[0]
        # make a backup next to the cache
        bak = OCR_CACHE / (cache.stem + ".bak.pdf")
        try:
            shutil.copy(pdf, bak)
        except Exception as e:
            print(f"skip (backup failed {pdf}): {e}")
            continue
        text = cache.read_text(encoding="utf-8", errors="replace")
        pages = parse_ocr_text(text)
        if not pages:
            print(f"skip (no parsed pages): {cache.name}")
            continue
        try:
            n, added = add_text_layer(str(pdf), pages)
            print(f"OK {pdf}  ({added}/{n} pages layered)")
        except Exception as e:
            print(f"ERR {pdf}: {e}")
            # restore backup
            shutil.copy(bak, pdf)


if __name__ == "__main__":
    main()