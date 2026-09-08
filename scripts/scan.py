#!/usr/bin/env python3
"""scan.py - Walk a directory, classify documents, report OCR needs and fiction tags."""

import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

EXTENSIONS_TEXT = {".pdf", ".epub", ".mobi", ".djvu", ".txt", ".html", ".htm", ".odt", ".docx", ".doc"}
OCR_BACKEND = "pdftotext"


def load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with open(cfg_path) as f:
        return json.load(f)


def calibre_db_path(target: Path) -> Path | None:
    """Find Calibre metadata.db by walking up from target, then searching within."""
    p = target
    while p != p.parent:
        db = p / "metadata.db"
        if db.exists():
            return db
        p = p.parent
    for db in target.rglob("metadata.db"):
        return db
    return None


def load_calibre_tags(db_path: Path) -> dict[str, list[str]]:
    """Return {filename: [tag_names]} from Calibre metadata.db."""
    tags_map: dict[str, list[str]] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("""
            SELECT d.name, d.format, GROUP_CONCAT(t.name, '|||')
            FROM data d
            JOIN books b ON d.book = b.id
            LEFT JOIN books_tags_link btl ON btl.book = b.id
            LEFT JOIN tags t ON btl.tag = t.id
            WHERE d.format IN ('PDF', 'EPUB', 'MOBI', 'DJVU', 'TXT')
            GROUP BY d.id
        """)
        for row in cur.fetchall():
            dname, fmt, tag_str = row
            tags = [t.strip() for t in tag_str.split("|||") if t.strip()] if tag_str else []
            tags_map[dname] = tags
            if fmt:
                tags_map[f"{dname}.{fmt.lower()}"] = tags
                tags_map[f"{dname}.{fmt.upper()}"] = tags
        conn.close()
    except Exception as e:
        console.print(f"[yellow]Warning: Could not read Calibre DB at {db_path}: {e}[/yellow]")
    return tags_map


def load_opf_tags(root: Path) -> dict[str, list[str]]:
    """Fallback: read metadata.opf files for tags."""
    tags_map: dict[str, list[str]] = {}
    for opf in root.rglob("metadata.opf"):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(opf)
            ns = {"dc": "http://purl.org/dc/elements/1.1/"}
            subjects = tree.findall(".//dc:subject", ns)
            tags = [s.text for s in subjects if s.text]
            parent_dir = opf.parent
            # Find the main file in the same directory
            for f in parent_dir.iterdir():
                if f.is_file() and f.suffix.lower() in EXTENSIONS_TEXT:
                    rel = str(f.relative_to(root))
                    tags_map[rel] = tags
                    break
        except Exception:
            pass
    return tags_map


def classify_kind(tags: list[str], fiction_tags: list[str]) -> str:
    """Classify fiction/nonfiction from Calibre tags."""
    tag_set = {t.lower() for t in tags}
    fiction_set = {t.lower() for t in fiction_tags}
    if tag_set & fiction_set:
        return "fiction"
    return "nonfiction"


def has_text_layer(pdf_path: str, pages: int = 3) -> tuple[bool, int]:
    """Check if a PDF has a text layer (pdftotext char count)."""
    try:
        result = subprocess.run(
            ["pdftotext", "-l", str(pages), pdf_path, "-"],
            capture_output=True, text=True, timeout=15
        )
        char_count = len(result.stdout.strip())
        return char_count > 50, char_count
    except Exception:
        return False, 0


def extract_epub_meta(epub_path: str) -> dict:
    """Quick metadata extraction from EPUB."""
    try:
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(epub_path)
        title = book.get_metadata("DC", "title")
        title = title[0][0] if title else Path(epub_path).stem
        return {"title": title}
    except Exception:
        return {"title": Path(epub_path).stem}


def scan_directory(target: str, cfg: dict, return_results: bool = False):
    target_path = Path(target)
    if not target_path.is_dir():
        console.print(f"[red]Error: {target} is not a directory[/red]")
        if return_results:
            raise FileNotFoundError(f"{target} is not a directory")
        sys.exit(1)

    fiction_tags = cfg.get("fiction_tags", ["Fiction", "Short Stories", "Literary"])
    ocr_threshold = cfg.get("ocr_char_threshold", 50)

    # Load Calibre metadata
    cal_db = calibre_db_path(target_path)
    cal_tags = {}
    if cal_db:
        console.print(f"[dim]Found Calibre DB: {cal_db}[/dim]")
        cal_tags = load_calibre_tags(cal_db)
    else:
        console.print("[yellow]No Calibre metadata.db found; using OPF fallback[/yellow]")
        cal_tags = load_opf_tags(target_path)

    # Scan files
    stats = defaultdict(lambda: {"count": 0, "ocr_needed": 0, "size": 0})
    fiction_count = 0
    nonfiction_count = 0
    untagged_count = 0
    ocr_files = []
    all_files = []
    seen_files = 0

    def progress(msg: str):
        sys.stderr.write(f"\r{msg}\x1b[K")
        sys.stderr.flush()

    last_dir = None
    for root, dirs, files in os.walk(target_path):
        # Skip hidden dirs and common non-content dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        if root != last_dir:
            rel_dir = str(Path(root).relative_to(target_path))
            progress(f"scanning: {rel_dir} ({seen_files} files seen)")
            last_dir = root

        dir_count = 0
        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext not in EXTENSIONS_TEXT:
                continue

            seen_files += 1
            dir_count += 1
            if dir_count % 10 == 0:
                progress(f"scanning: {Path(root).relative_to(target_path)} "
                         f"[{seen_files} files seen]")

            rel = str(fpath.relative_to(target_path))
            size = fpath.stat().st_size
            stats[ext]["count"] += 1
            stats[ext]["size"] += size

            info = {
                "path": rel,
                "full_path": str(fpath),
                "ext": ext,
                "size": size,
                "kind": "unknown",
                "needs_ocr": False,
                "char_count": 0,
                "tags": [],
                "title": fpath.stem,
            }

            # Get Calibre tags
            # Calibre stores paths as "Author/Title (ID)/filename"
            cal_key = None
            for ck in cal_tags:
                if fname in ck or ck.endswith(fname):
                    cal_key = ck
                    break
            if cal_key:
                info["tags"] = cal_tags[cal_key]

            # OPF fallback
            if not info["tags"]:
                opf_tags = load_opf_tags(fpath.parent)
                if rel in opf_tags:
                    info["tags"] = opf_tags[rel]

            # Classify fiction
            info["kind"] = classify_kind(info["tags"], fiction_tags)
            if info["kind"] == "fiction":
                fiction_count += 1
            else:
                if not info["tags"]:
                    untagged_count += 1
                else:
                    nonfiction_count += 1

            # Check for OCR needs (PDF only)
            if ext == ".pdf":
                has_text, chars = has_text_layer(str(fpath))
                info["char_count"] = chars
                if not has_text:
                    info["needs_ocr"] = True
                    stats[ext]["ocr_needed"] += 1
                    ocr_files.append(info)

            all_files.append(info)

    progress(f"done: {seen_files} files seen")
    sys.stderr.write("\n")
    sys.stderr.flush()

    # Summary table
    if return_results:
        return build_results(target_path, all_files, cal_tags, ocr_files,
                             fiction_count, nonfiction_count, untagged_count, stats)

    console.print()
    console.print(f"[bold]Scan complete: {target_path}[/bold]")
    console.print(f"[dim]Calibre tags loaded for {len(cal_tags)} books[/dim]")
    console.print()

    table = Table(title="File Inventory", box=box.ROUNDED)
    table.add_column("Extension", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("OCR Needed", justify="right", style="yellow")
    table.add_column("Size", justify="right")

    for ext in sorted(stats.keys()):
        s = stats[ext]
        size_mb = s["size"] / (1024 * 1024)
        table.add_row(
            ext,
            str(s["count"]),
            str(s["ocr_needed"]) if s["ocr_needed"] else "-",
            f"{size_mb:.1f} MB",
        )
    console.print(table)

    # Classification summary
    console.print()
    console.print(f"[green]Fiction: {fiction_count}[/green]  "
                  f"[blue]Non-Fiction: {nonfiction_count}[/blue]  "
                  f"[yellow]Untagged (default nonfiction): {untagged_count}[/yellow]")

    # Fiction files list
    fiction_files = [f for f in all_files if f["kind"] == "fiction"]
    if fiction_files:
        console.print()
        console.print("[bold green]Fiction works:[/bold green]")
        for f in fiction_files:
            tags_str = ", ".join(f["tags"]) if f["tags"] else "no tags"
            console.print(f"  {f['path']}  [dim]({tags_str})[/dim]")

    # OCR files list
    if ocr_files:
        console.print()
        console.print(f"[bold yellow]Files needing OCR ({len(ocr_files)}):[/bold yellow]")
        for f in ocr_files:
            console.print(f"  {f['path']}")

    # Persist JSON for downstream use (runs when console mode, not return_results)
    build_results(target_path, all_files, cal_tags, ocr_files,
                  fiction_count, nonfiction_count, untagged_count, stats)
    console.print(f"\n[dim]Full results written to "
                  f"{Path(__file__).resolve().parent.parent / 'scan_results.json'}[/dim]")


def build_results(target_path, all_files, cal_tags, ocr_files,
                  fiction_count, nonfiction_count, untagged_count, stats):
    """Return scan results as a dict (used by web app) and persist to JSON."""
    result = {
        "target": str(target_path),
        "total_files": len(all_files),
        "fiction_count": fiction_count,
        "nonfiction_count": nonfiction_count,
        "untagged_count": untagged_count,
        "ocr_needed": len(ocr_files),
        "stats": {
            ext: {"count": s["count"], "ocr_needed": s["ocr_needed"], "size": s["size"]}
            for ext, s in stats.items()
        },
        "files": all_files,
    }
    out_path = Path(__file__).resolve().parent.parent / "scan_results.json"
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2, default=str)
    return result


if __name__ == "__main__":
    cfg = load_config()
    target = sys.argv[1] if len(sys.argv) > 1 else cfg["sets"]["library"]["path"]
    scan_directory(target, cfg)
