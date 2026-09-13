#!/usr/bin/env python3
"""ocr_compare.py - Compare Tesseract vs RapidOCR on a 5-page sample of every
PDF that needs OCR, pick the better engine, then OCR the full file with it.

Flow per file:
  1. Render first N pages to PNG (PyMuPDF @200dpi).
  2. OCR the sample with both Tesseract and RapidOCR.
  3. Score "well-formedness" = weighted confidence - garbage.
  4. Route: Tesseract if its sample quality >= tess_threshold (fast default);
     else RapidOCR if its quality >= rapid_threshold; else flag too_bad.
  5. OCR the full file with the chosen engine -> ocr/<stem>.txt cache
     (the path ingest.py already reads). Resumable: files with an existing
     cache (and non-zero size) are skipped unless --force.

Report: ocr_compare_report.json + rich table.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from rich.console import Console
from rich.table import Table
from rich import box

try:
    import pytesseract
except ImportError:
    pytesseract = None

console = Console()

from _paths import rag_root

RAG_ROOT = rag_root()
SAMPLE_ROOT = Path(os.environ.get("OCR_SAMPLE_DIR", "/tmp/opencode/ocr_samples"))
OCR_CACHE = RAG_ROOT / "ocr"
REPORT_PATH = RAG_ROOT / "ocr_compare_report.json"
DEFAULT_TESS_BIN = "/tmp/opencode/mamba/envs/tess/bin/tesseract"
DEFAULT_TESSDATA = "/tmp/opencode/mamba/envs/tess/share/tessdata"

# Words/confidences below these are junk regardless of structure.
LANG_MAP = {"en": "eng", "es": "spa", "de": "deu", "fr": "fra", "it": "ita", "pt": "por", "ru": "rus"}

PUNCT_OK = set(".,;:!?'\"-—–()[]{}/*\\%$&+=<>@#|~_`^")
ALNUM_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def load_config():
    with open(RAG_ROOT / "config.json") as f:
        return json.load(f)


def needs_ocr(pdf_path: str, threshold: int = 50) -> bool:
    try:
        result = subprocess.run(
            ["pdftotext", "-l", "3", pdf_path, "-"],
            capture_output=True, text=True, timeout=15,
        )
        return len(result.stdout.strip()) < threshold
    except Exception:
        return True


def _setup_tesseract(bin_path: str | None, tessdata: str | None):
    if pytesseract is None:
        return False
    bin_path = (
        bin_path
        or os.environ.get("TESSERACT_BIN")
        or shutil.which("tesseract")
        or DEFAULT_TESS_BIN
    )
    if not os.path.exists(bin_path):
        return False
    pytesseract.pytesseract.tesseract_cmd = bin_path
    if tessdata and os.path.isdir(tessdata):
        os.environ["TESSDATA_PREFIX"] = tessdata
    return True


def _tess_lang(languages):
    out = []
    for l in (languages or ["en"]):
        out.append(LANG_MAP.get(l, l if l != "en" else "eng"))
    return "+".join(out)


def discover(target: str, threshold: int) -> list[dict]:
    """Return file infos needing OCR (skipping encrypted + text-layer PDFs)."""
    target_path = Path(target)
    infos = []
    for fpath in sorted(target_path.rglob("*.pdf")):
        if fpath.name.startswith("."):
            continue
        rel = str(fpath.relative_to(target_path))
        try:
            doc = fitz.open(str(fpath))
            if doc.needs_pass:
                infos.append({"rel": rel, "full": str(fpath), "encrypted": True,
                              "needs_ocr": False, "pages": doc.page_count,
                              "size": fpath.stat().st_size})
                doc.close()
                continue
            pages = doc.page_count
            doc.close()
        except Exception:
            infos.append({"rel": rel, "full": str(fpath), "encrypted": True,
                          "needs_ocr": False, "pages": -1, "size": fpath.stat().st_size})
            continue
        infos.append({"rel": rel, "full": str(fpath), "encrypted": False,
                      "needs_ocr": needs_ocr(str(fpath), threshold),
                      "pages": pages, "size": fpath.stat().st_size})
    return infos


def render_pages(pdf_path: str, n: int, dpi: int = 200, out_dir: Path | None = None) -> list[str]:
    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        doc.close()
        return []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    base = Path(pdf_path).stem
    for i in range(min(n, len(doc))):
        pix = doc[i].get_pixmap(matrix=mat)
        if out_dir is None:
            out_dir = SAMPLE_ROOT / Path(pdf_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{base}_p{i:03d}.png"
        pix.save(str(p))
        paths.append(str(p))
    doc.close()
    return paths


# ─── Engine calls ────────────────────────────────────────────────────────────

def tess_page(png_path: str, lang: str) -> tuple[str, float]:
    try:
        data = pytesseract.image_to_data(png_path, lang=lang, output_type=pytesseract.Output.DICT)
        texts = data.get("text", []) or []
        confs = data.get("conf", []) or []
        words, confs_ok = [], []
        for t, c in zip(texts, confs):
            t = str(t or "").strip()
            if not t:
                continue
            words.append(t)
            # tesseract sets conf=-1 for tokens it could not classify as a word
            # (stray symbols / fragments). Count those as 0 confidence so
            # garbage output lowers the page mean.
            try:
                cc = max(float(c), 0.0)
            except (TypeError, ValueError):
                cc = 0.0
            confs_ok.append(cc)
        text = "\n".join(words)
        conf = sum(confs_ok) / len(confs_ok) / 100.0 if confs_ok else 0.0
        return text, conf
    except Exception:
        return "", 0.0


def rapid_page(engine, png_path: str) -> tuple[str, float]:
    try:
        result, _ = engine(png_path)
        if not result:
            return "", 0.0
        lines = [r[1] for r in result if len(r) > 1 and r[1]]
        scores = [float(r[2]) for r in result if len(r) > 2 and r[2] is not None]
        text = "\n".join(lines)
        conf = sum(scores) / len(scores) if scores else 0.0
        return text, conf
    except Exception:
        return "", 0.0


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _garbage_ratio(text: str) -> float:
    """Fraction of characters/tokens that look like OCR junk.

    Three signals:
      - control/replacement/non-whitelist characters
      - stray short tokens (symbols, single chars, no-vowel fragments)
      - tokens that are mostly non-alphanumeric
    """
    if not text or not text.strip():
        return 1.0
    total_chars = max(len(text), 1)
    garb_chars = 0
    foreign_tokens = 0
    for ch in text:
        o = ord(ch)
        if ch == "\ufffd" or (o < 32 and ch not in "\n\t\r"):
            garb_chars += 1
        elif ch.isalnum() or ch.isspace() or ch in PUNCT_OK or ch in "€£¥°¿¿¡ªº®™":
            continue
        else:
            garb_chars += 1

    tokens = text.split()
    half_alpha = 0
    no_vowel = 0
    single_or_sym = 0
    for tok in tokens[:800]:
        if not tok:
            continue
        alnum = sum(1 for c in tok if c.isalnum())
        alpha_ratio = alnum / len(tok)
        if alpha_ratio < 0.6:
            half_alpha += 1
        elif tok.isalpha():
            low = tok.lower()
            if len(tok) == 1 and low not in ("a", "i"):
                single_or_sym += 1
            # 2+ char alpha token with no vowel is almost certainly OCR garbage
            elif len(tok) >= 2 and not any(v in low for v in "aeiouy"):
                no_vowel += 1

    n_tok = max(len(tokens), 1)
    tok_junk = (half_alpha + no_vowel + single_or_sym) / n_tok
    char_junk = garb_chars / total_chars
    # token junk is the stronger signal; char junk catches control chars too
    return min(0.7 * tok_junk + 0.3 * char_junk, 1.0)


def composite(conf: float, garbage: float, w_conf: float = 0.6, w_clean: float = 0.4) -> float:
    return w_conf * conf + w_clean * (1.0 - garbage)


def sample_score(pngs: list[str], lang: str, engine=None):
    """Run an engine over sample pages; return (pages_text, mean_conf, garbage, quality)."""
    texts, confs = [], []
    for p in pngs:
        if engine is not None:
            t, c = rapid_page(engine, p)
        else:
            t, c = tess_page(p, lang)
        texts.append(t)
        confs.append(c)
    text = "\n\n".join(texts)
    n_conf = sum(confs) / len(confs) if confs else 0.0
    garb = _garbage_ratio(text)
    return text, n_conf, garb, composite(n_conf, garb)


# ─── Full-file OCR (parallel page workers) ───────────────────────────────────

from concurrent.futures import ThreadPoolExecutor, as_completed


def _page_tess_full(pdf_path: str, page_num: int, lang: str, dpi: int) -> str:
    """Render one page and OCR it with tesseract; return page text."""
    import tempfile
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        pix.save(tmp)
    finally:
        doc.close()
    try:
        return pytesseract.image_to_string(tmp, lang=lang)
    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _page_rapid_full(engine, pdf_path: str, page_num: int, dpi: int) -> str:
    import tempfile
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        pix.save(tmp)
    finally:
        doc.close()
    try:
        result, _ = engine(tmp)
        return "\n".join(r[1] for r in result if len(r) > 1 and r[1]) if result else ""
    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _ocr_full_parallel(pdf_path: str, out_path: Path, page_task, workers: int,
                       on_page=None) -> bool:
    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        doc.close()
        return False
    n = len(doc)
    doc.close()

    results = [""] * n
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(page_task, i): i for i in range(n)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result(timeout=120)
            except Exception:
                results[i] = ""
            done += 1
            if on_page and (done % 10 == 0 or done == n):
                on_page(done, n)

    all_text = [f"--- Page {i + 1} ---\n{results[i]}" for i in range(n)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(all_text), encoding="utf-8")
    return bool(any(results))


def ocr_full_tesseract(pdf_path: str, lang: str, out_path: Path, on_page=None,
                       workers: int = 8) -> bool:
    def task(i):
        return _page_tess_full(pdf_path, i, lang, 200)
    return _ocr_full_parallel(pdf_path, out_path, task, workers, on_page)


def ocr_full_rapidocr(engine, pdf_path: str, out_path: Path, on_page=None,
                      workers: int = 4) -> bool:
    def task(i):
        return _page_rapid_full(engine, pdf_path, i, 200)
    return _ocr_full_parallel(pdf_path, out_path, task, workers, on_page)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare OCR engines, then full-OCR with the winner")
    ap.add_argument("target", help="Directory to scan (e.g. /media/veracrypt1/Private Library)")
    ap.add_argument("--only", help="Only process paths containing this substring")
    ap.add_argument("--sample", type=int, default=5, help="Sample pages per file")
    ap.add_argument("--limit", type=int, default=0, help="Max files to process (test mode)")
    ap.add_argument("--sample-only", action="store_true",
                    help="Only run the 5-page comparison; skip full-file OCR")
    ap.add_argument("--force", action="store_true", help="Re-OCR files that already have a cache")
    ap.add_argument("--force-tesseract", action="store_true",
                    help="Force tesseract for all files regardless of routing")
    ap.add_argument("--tess-threshold", type=float, default=0.50)
    ap.add_argument("--rapid-threshold", type=float, default=0.45)
    ap.add_argument("--workers", type=int, default=8, help="Parallel OCR page workers")
    ap.add_argument("--tesseract", default=None, help="Path to tesseract binary")
    ap.add_argument("--tessdata", default=None, help="Path to tessdata dir")
    args = ap.parse_args()

    if not _setup_tesseract(args.tesseract, args.tessdata):
        console.print("[red]pytesseract or tesseract binary not available. "
                      "Install pytesseract and tesseract-ocr. TESSERACT_BIN env can override.[/red]")
        sys.exit(1)

    cfg = load_config()
    languages = cfg.get("ocr_languages", ["en"])
    threshold = cfg.get("ocr_char_threshold", 50)

    try:
        from rapidocr_onnxruntime import RapidOCR
        rapid = RapidOCR()
    except ImportError:
        console.print("[yellow]rapidocr-onnxruntime not installed; rapidocr route unavailable[/yellow]")
        rapid = None

    target = Path(args.target)
    if not target.is_dir():
        console.print(f"[red]{target} is not a directory[/red]")
        sys.exit(1)

    OCR_CACHE.mkdir(exist_ok=True)
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)

    infos = discover(str(target), threshold)
    work = [i for i in infos if not i["encrypted"] and i["needs_ocr"]]
    skipped_enc = [i for i in infos if i["encrypted"]]
    console.print(f"[dim]{len(infos)} PDFs total | "
                  f"{len(work)} need OCR | {len(skipped_enc)} encrypted (skipped)[/dim]")
    if args.only:
        work = [i for i in work if args.only in i["rel"]]
    if args.limit:
        work = work[: args.limit]
    if not work:
        console.print("[green]Nothing to do.[/green]")
        return

    # Load previous report for resume
    report = {}
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text())
        except Exception:
            report = {}
    report["decisions"] = report.get("decisions", {})

    lang = _tess_lang(languages)
    console.print(f"[bold]{len(work)} files to compare (tesseract lang={lang}, "
                  f"sample={args.sample} pages)[/bold]\n")

    t0 = time.time()
    done = 0
    for idx, fi in enumerate(work, 1):
        rel = fi["rel"]
        stem = Path(rel).stem
        cache = OCR_CACHE / (stem + ".txt")
        if cache.exists() and cache.stat().st_size > 0 and not args.force:
            report["decisions"][rel] = {
                "pages": fi["pages"], "size": fi["size"],
                "decision": "cached", "reason": "existing ocr cache",
                "tess_quality": None, "rapid_quality": None,
            }
            done += 1
            continue

        elapsed = int(time.time() - t0)
        print(f"[{idx}/{len(work)}] {stem} ({fi['pages']}p, {fi['size']//1048576}MB)"
              f" elapsed={elapsed//60}m{elapsed%60}s", flush=True)

        pngs = render_pages(fi["full"], args.sample)
        if not pngs:
            report["decisions"][rel] = {"pages": fi["pages"], "size": fi["size"],
                                        "decision": "error", "reason": "render failed"}
            continue

        t_text, t_conf, t_garb, t_q = sample_score(pngs, lang)
        r_text, r_conf, r_garb, r_q = (None, 0, 1.0, 0.0)
        if rapid is not None:
            r_text, r_conf, r_garb, r_q = sample_score(pngs, lang, engine=rapid)

        print(f"    tesseract: q={t_q:.2f} (conf={t_conf:.2f}, garb={t_garb:.2f}) "
              f"[{t_text.strip()[:60]!r}]")
        if rapid is not None:
            print(f"    rapidocr : q={r_q:.2f} (conf={r_conf:.2f}, garb={r_garb:.2f}) "
                  f"[{r_text.strip()[:60]!r}]")

        if (t_q >= args.tess_threshold
                and (rapid is None or (r_q - t_q) <= 0.10)):
            decision, reason = "tesseract", f"tess_q={t_q:.2f} >= {args.tess_threshold} (rapid {r_q:.2f})"
        elif rapid is not None and r_q >= args.rapid_threshold:
            decision, reason = "rapidocr", f"tess bad/worse ({t_q:.2f} vs {r_q:.2f}); rapid_q >= {args.rapid_threshold}"
        else:
            decision, reason = "too_bad", (f"tess_q={t_q:.2f}, rapid_q={r_q:.2f} "
                                           "below thresholds")
        report["decisions"][rel] = {
            "pages": fi["pages"], "size": fi["size"],
            "decision": decision, "reason": reason,
            "tess_quality": round(t_q, 3), "rapid_quality": round(r_q, 3),
            "tess_conf": round(t_conf, 3), "rapid_conf": round(r_conf, 3),
            "tess_garbage": round(t_garb, 3), "rapid_garbage": round(r_garb, 3),
            "sample_preview": (t_text.strip()[:120] if decision == "tesseract"
                               else r_text.strip()[:120] if decision == "rapidocr" else t_text.strip()[:120]),
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2))

        if args.sample_only:
            fdir = SAMPLE_ROOT / stem
            if fdir.is_dir():
                for p in fdir.glob("*.png"):
                    p.unlink(missing_ok=True)
            continue

        if decision == "too_bad" and not args.force_tesseract:
            print(f"    -> {decision} ({reason})", flush=True)
            continue

        if args.force_tesseract:
            decision = "tesseract"
            reason = "forced tesseract"
        print(f"    -> full OCR with {decision}...", flush=True)
        ok = False
        if decision == "tesseract":
            ok = ocr_full_tesseract(fi["full"], lang, cache,
                                    on_page=lambda pg, tot: print(
                                        f"    [tesseract] page {pg}/{tot}", end="\r", flush=True)
                                    if pg % 15 == 0 or pg == tot else None,
                                    workers=args.workers)
        elif rapid is not None:
            ok = ocr_full_rapidocr(rapid, fi["full"], cache,
                                   on_page=lambda pg, tot: print(
                                       f"    [rapidocr] page {pg}/{tot}", end="\r", flush=True)
                                   if pg % 15 == 0 or pg == tot else None,
                                   workers=max(1, args.workers // 2))
        print()
        if ok:
            report["decisions"][rel]["cache"] = str(cache)
            report["decisions"][rel]["status"] = "ocred"
        else:
            report["decisions"][rel]["status"] = "ocr_failed"
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        done += 1
        # cleanup sample pngs for this file
        fdir = SAMPLE_ROOT / stem
        if fdir.is_dir():
            for p in fdir.glob("*.png"):
                p.unlink(missing_ok=True)

    # ─── Summary table ───────────────────────────────────────────────────
    dec = report["decisions"]
    counts = {}
    for rel, d in dec.items():
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1
    console.print()
    console.print(f"[bold]OCR comparison complete ({done} files this run)[/bold]")
    console.print(f"  Decisions: {dict(counts)}")
    console.print(f"  Report: {REPORT_PATH}")

    table = Table(title="OCR engine routing", box=box.ROUNDED)
    table.add_column("Decision", style="bold")
    table.add_column("Count", justify="right")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        table.add_row(k, str(v))
    console.print(table)

    if counts.get("too_bad"):
        console.print("\n[bold yellow]Too bad for either engine:[/bold yellow]")
        for rel, d in sorted(dec.items()):
            if d["decision"] == "too_bad":
                console.print(f"  {rel}  [dim]({d['reason']})[/dim]")

    report["_summary"] = {"counts": counts, "total_pages": sum(d["pages"] for d in dec.values() if d["pages"] > 0)}
    REPORT_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()