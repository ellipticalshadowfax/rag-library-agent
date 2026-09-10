#!/usr/bin/env python3
"""eval.py - deterministic offline eval harness for the retrieval layer.

Scores retrieval only (no LLM calls). Ground truth is "did the expected
source surface in the top-K / context", derived from known titles in
manifest.db via evals/golden.jsonl.

Subcommands:
  run       run retrieval over the golden set and write evals/results.json
  baseline  snapshot evals/results.json -> evals/baseline.json
  diff      compare evals/results.json vs evals/baseline.json; exit nonzero
            if any metric drops more than --tolerance (CI regression gate)

Note: multi-hop retrieval is disabled here (client=None) because it needs an
LLM; this keeps the harness deterministic and offline.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import agent

from _paths import rag_root

RAG_ROOT = rag_root()
EVALS_DIR = RAG_ROOT / "evals"
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
RESULTS_PATH = EVALS_DIR / "results.json"
BASELINE_PATH = EVALS_DIR / "baseline.json"
METRIC_KEYS = ["hit@5", "hit@10", "mrr@5", "mrr@10",
               "context_recall", "context_precision"]

_embedder = {"obj": None}
_reranker = {"obj": None, "model": None}
_rerank_unavailable = set()
_collections = {}


def _load_golden(set_filter=None):
    items = []
    with open(GOLDEN_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if set_filter and item.get("set") != set_filter:
                continue
            items.append(item)
    return items


def _get_embedder(cfg):
    if _embedder["obj"] is None:
        _embedder["obj"] = agent.setup_embedder(cfg)
    return _embedder["obj"]


def _get_reranker(cfg):
    if not cfg.get("rerank_enabled", True):
        return None
    model = cfg.get("rerank_model")
    if not model or model in _rerank_unavailable:
        return None
    if _reranker["model"] != model or _reranker["obj"] is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"[eval] loading reranker: {model}...")
            _reranker["obj"] = CrossEncoder(
                model, device=cfg.get("embed_device", "cpu"))
            _reranker["model"] = model
        except Exception as e:
            print(f"[eval] reranker unavailable, skipping rerank: {e}")
            _rerank_unavailable.add(model)
            _reranker["obj"] = None
            return None
    return _reranker["obj"]


def _get_collection(set_name):
    if set_name in _collections:
        return _collections[set_name]
    try:
        _collections[set_name] = agent.setup_chroma(set_name)
    except SystemExit:
        raise ValueError(f"collection '{set_name}' not found (run ingest first)")
    return _collections[set_name]


def _title_matches(hit_title, expected_title):
    a = (hit_title or "").strip().lower()
    b = (expected_title or "").strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


def _compute_metrics(result, expected_title):
    hits = result["hits"]
    mets = {}
    for k in (5, 10):
        mets[f"hit@{k}"] = 0.0
        mets[f"mrr@{k}"] = 0.0
        for i, h in enumerate(hits[:k]):
            if _title_matches(h.get("metadata", {}).get("title"), expected_title):
                mets[f"hit@{k}"] = 1.0
                mets[f"mrr@{k}"] = 1.0 / (i + 1)
                break

    # context_recall: does the expected source's text appear in the context
    # (the final, budget-trimmed hits that built message_context).
    mets["context_recall"] = float(any(
        _title_matches(h.get("metadata", {}).get("title"), expected_title)
        for h in hits))

    # context_precision: over the deduped (title, source) sources shown to the
    # user, what fraction match the expected source.
    sources = result["sources"]
    mets["context_precision"] = (
        sum(1.0 for s in sources
            if _title_matches(s.get("title"), expected_title)) / len(sources)
        if sources else 0.0)
    return mets


def _evaluate_query(item, top_k, cfg):
    set_name = item["set"]
    result = agent.retrieve_rag(
        item["set"], item["query"], top_k, item.get("filter_kind"),
        cfg, _get_embedder(cfg), _get_collection(set_name),
        client=None, reranker=_get_reranker(cfg))
    return result


def _aggregate(query_records):
    agg = {}
    for key in METRIC_KEYS:
        vals = [q["metrics"][key] for q in query_records]
        agg[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return agg


def cmd_run(args):
    if not GOLDEN_PATH.exists():
        print(f"[eval] no golden set at {GOLDEN_PATH} - create it first")
        return 0
    items = _load_golden(args.set)
    if not items:
        print("[eval] no golden queries match --set filter")
        return 0
    cfg = agent.load_config()
    try:
        _get_embedder(cfg)
    except Exception as e:
        print(f"[eval] embedder unavailable, skipping run: {e}")
        return 0

    records = []
    for i, item in enumerate(items, 1):
        q, expected, s = item["query"], item["expected_title"], item["set"]
        try:
            result = _evaluate_query(item, args.top_k, cfg)
            mets = _compute_metrics(result, expected)
        except Exception as e:
            print(f"[eval] [{s}] SKIP '{expected}': {e}")
            continue
        records.append({
            "query": q, "expected_title": expected, "set": s,
            "filter_kind": item.get("filter_kind"),
            "metrics": mets,
            "retrieval": {
                "num_hits": len(result["hits"]),
                "title_mode": result["title_mode"],
                "matched_titles": result["matched_titles"],
                "fiction_only": result["fiction_only"],
                "low_relevance": result["low_relevance"],
                "relevance_reason": result["relevance_reason"],
            },
        })
        print(f"[eval] ({i}/{len(items)}) [{s}] hit@5={mets['hit@5']}"
              f" mrr@5={mets['mrr@5']} ctx_recall={mets['context_recall']}"
              f" ctx_prec={mets['context_precision']}"
              f" | {expected}")
        if result["low_relevance"]:
            print(f"    low-relevance: {result['relevance_reason']}")
        if result["title_mode"]:
            print(f"    title-mode -> {result['matched_titles']}")

    if not records:
        print("[eval] no queries evaluated")
        return 1

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "num_queries": len(records),
        "sets": sorted({r["set"] for r in records}),
        "metrics": _aggregate(records),
        "queries": records,
    }
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print("\n[eval] summary:")
    for key, val in summary["metrics"].items():
        print(f"    {key:20s} {val:.4f}")
    print(f"[eval] wrote {RESULTS_PATH}")
    return 0


def cmd_baseline(args):
    if not RESULTS_PATH.exists():
        print("[eval] no results.json - run 'eval.py run' first")
        return 1
    shutil.copyfile(RESULTS_PATH, BASELINE_PATH)
    print(f"[eval] baseline written to {BASELINE_PATH}")
    return 0


def cmd_diff(args):
    if not RESULTS_PATH.exists() or not BASELINE_PATH.exists():
        print("[eval] missing results.json and/or baseline.json")
        return 1
    cur = json.loads(RESULTS_PATH.read_text())
    base = json.loads(BASELINE_PATH.read_text())
    tol = max(float(args.tolerance), 0.0)
    failed = False
    print("[eval] diff vs baseline (tolerance "
          f"{tol:.0%}):")
    for key in METRIC_KEYS:
        c, b = cur["metrics"].get(key, 0.0), base["metrics"].get(key, 0.0)
        regressed = c < b * (1 - tol)
        failed = failed or regressed
        print(f"    {key:20s} baseline {b:.4f}  current {c:.4f}"
              f"  {'REGRESSION' if regressed else 'ok'}")
    if failed:
        print("[eval] FAIL: metric(s) dropped by more than "
              f"{tol:.0%}")
        return 1
    print("[eval] PASS: no regression vs baseline")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="Offline retrieval eval harness (no LLM calls).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run golden-set evaluation")
    p_run.add_argument("--set", default=None, help="only this library set")
    p_run.add_argument("--top-k", type=int, default=10,
                       help="retrieval top-k (default 10)")
    p_run.set_defaults(func=cmd_run)

    p_bl = sub.add_parser("baseline", help="snapshot results -> baseline")
    p_bl.set_defaults(func=cmd_baseline)

    p_df = sub.add_parser("diff", help="compare results vs baseline")
    p_df.add_argument("--tolerance", type=float, default=0.05,
                      help="max allowed per-metric drop (default 0.05)")
    p_df.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()