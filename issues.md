# Known Issues

Status legend: **OPEN** (still a real problem) · **MINOR** (mostly fixed; small
edge case remains) · **FIXED** (resolved in current code).

## 1. Unexpected full reingest of a collection (not incremental) — OPEN

Updating/cloning collection `veracrypt1` via the web UI sometimes reprocesses the
entire library instead of only changed files. The web UI does **not** pass
`--force` by default, so a full reingest is caused by one of the following:

- **Stat-only change detection** (`scripts/ingest.py:1119-1124`). There is no
  content-hash check — only `(rel_path, mtime, size)`. Copy/restore of the
  library without `-a`, or tools that rewrite files wholesale, touch every file
  and trigger a full reindex. Still the primary residual cause.
- **`manifest.db` missing/wiped**. A fresh `RAG_ROOT` or a deleted manifest
  empties the manifest so `is_current()` returns `False` for all files. Inherent
  to stat-based change detection.
- **Target path changed between runs** (`scripts/ingest.py:1071-1081`). Change
  detection keys on `rel_path`, so picking a *different* parent/subdirectory in
  the folder picker makes every `rel_path` new and every file looks "changed".
  (Mitigated: the ingest now records `meta.target:<set>` and warns when the
  target changes; a pure mount-point change with an identical relative tree is
  still skipped.)

The set-agnostic bookkeeping cause (previously issue 3) is **fixed**: change
detection is now scoped per set (see #3).

## 2. OCR `ocr_merge` causes repeat reprocessing of all scanned PDFs — MINOR

`merge_text_into_pdf()` (`scripts/ingest.py:554`) modifies the source PDF in
place, changing its size/mtime. Fixed: after a successful merge the fresh
post-merge `mtime`/`size` are written back into the manifest
(`ingest.py:1233-1236,1369`), and the merge is gated on the OCR cache being
newer than the PDF (`ingest.py:1229-1239`), so a merged PDF is seen as current
and never re-merged.

Residual: `needs_ocr()` still returns `True` on any exception
(`scripts/ingest.py:487-488`), so if `pdftotext` is missing from PATH every PDF
is flagged as OCR-needing every run.

## 3. Manifest is not set-aware — FIXED

`files.rel_path` was `UNIQUE` across all sets; keys now include `set_name`:
`UNIQUE(rel_path, set_name)` (`scripts/ingest.py:118`) and
`PRIMARY KEY (parent_id, set_name)` for `parents` (`ingest.py:145`).
`_migrate_legacy_schemas()` (`ingest.py:157-240`) rebuilds legacy tables with
scoped keys; `is_current()` consults `set_name` (`ingest.py:242-248`); upserts
conflict on `(rel_path, set_name)`; and `_drop_manifest_set()`
(`server.py:3326-3341`) deletes only that set's rows, so deleting the row-owning
collection no longer wipes shared rows for other sets.

## 4. Stale chunks orphaned in Chroma after a path change — MINOR

Previously, stale-chunk cleanup deleted by the **new** rel path only
(`collection.delete(where={"source": rel})`), orphaning old-path chunks and
leaving dead manifest rows. Fixed: a prune pass (`scripts/ingest.py:1161-1183`)
walks this set's manifest rows and, for any rel path not seen in the walk whose
recorded abs path no longer exists, deletes the Chroma chunks, the manifest row,
and the parents row.

Residual edge case: pruning requires `not Path(abs_path).exists()`. If the set
is re-pointed to a different directory while the old files still exist on disk
(e.g. the old library is still mounted elsewhere), old rel paths fall outside
the prune condition and their chunks stay orphaned. Pruning also skips
`only_path` runs.

## 5. `agent.py` hardcodes the manifest path instead of using `rag_root()` — FIXED

All manifest accesses in `agent.py` (BM25 hydration, parent-text store, title
matching) now use `rag_root() / "manifest.db"` (`scripts/agent.py:251,677,857`),
matching `ingest.py`, `server.py`, and `mcp_server.py`. In a symlinked
deployment where data lives outside the code root, `agent.py` now resolves the
same manifest as the rest of the app.