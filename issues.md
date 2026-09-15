# Known Issues

## 1. Unexpected full reingest of a collection (not incremental)

Updating/cloning collection `veracrypt1` via the web UI sometimes reprocesses the
entire library instead of only changed files. The web UI does **not** pass
`--force` by default (`scripts/server.py:1087`, `web/index.html:1440,1468`), so a
full reingest is caused by one of the following:

- **Target path changed between runs** (`scripts/ingest.py:952,962`). Change
  detection keys on `(rel_path, mtime, size)`. If the Veracrypt volume is mounted
  at a different path between sessions, or a different parent/subdirectory is
  picked in the folder picker, every `rel_path` is new and every file looks
  "changed".
- **`manifest.db` missing/wiped** (`scripts/ingest.py:923`, `server.py:1719-1727`).
  A fresh `RAG_ROOT`, a deleted manifest, or a prior UI delete of the row-owning
  collection empties the manifest so `is_current()` returns `False` for all files.
- **Bulk mtime/size changes on disk**. There is no content-hash check — only
  stat-based. Copy/restore of the library without `-a`, or tools that rewrite
  files wholesale, touch every file and trigger a full reindex.
- **Set-agnostic manifest bookkeeping** (see issue 3 below).

## 2. OCR `ocr_merge` causes repeat reprocessing of all scanned PDFs

`merge_text_into_pdf()` (`scripts/ingest.py:1038-1039`) modifies the source PDF in
place, changing its size/mtime, but the manifest stores the **pre-merge** values
(`ingest.py:1170`). On the next ingest every merged PDF looks changed and is
reprocessed. Amplified further: `needs_ocr()` returns `True` on any exception
(`ingest.py:334-342`), so if `pdftotext` is missing from PATH every PDF is flagged
every run and the merge branch re-merges over existing cached OCR.

## 3. Manifest is not set-aware: `rel_path` globally unique, `is_current()` ignores `set_name`

`files.rel_path` is `UNIQUE` across all sets (`scripts/ingest.py:108`) and
`is_current()` never consults `set_name` (`ingest.py:149-153`). Consequences:

- Set B ingesting a directory already recorded by set A silently populates
  nothing (everything is "current") and bypasses the upsert.
- If set A (the row owner) is deleted, `_drop_manifest_set` deletes all shared
  rows → set B's next ingest does a full reingest.
- `ON CONFLICT(rel_path) DO UPDATE ... set_name=excluded.set_name`
  (`ingest.py:161-170`) means the set that ingested last "steals" ownership of
  every shared row.

## 4. Stale chunks orphaned in Chroma after a path change

The stale-chunk cleanup `collection.delete(where={"source": rel})`
(`scripts/ingest.py:1102-1106`) deletes by the **new** rel path only, so after a
target-path change the old-path chunks are orphaned in Chroma and the collection
grows with duplicates. The manifest likewise keeps dead rows for rel paths no
longer present under the target (no pruning).

## 5. `agent.py` hardcodes the manifest path instead of using `rag_root()`

`scripts/agent.py:525` (BM25 hydration) and `scripts/agent.py:688` (parent-text
store) use `Path(__file__).resolve().parent.parent / "manifest.db"`, while
`ingest.py`, `server.py`, and `mcp_server.py` all use `rag_root()`. In a
symlinked deployment where data lives outside the code root, `agent.py` reads a
different/nonexistent manifest → BM25 falls back to rebuilding from Chroma and
parent texts load as `{}`. Retrieval-side; not a reingest trigger.