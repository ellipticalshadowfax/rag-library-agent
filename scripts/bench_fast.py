#!/usr/bin/env python3
"""Bench fast CPU embedding models vs BGE-M3 on real indexed chunks."""
import time, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(16)
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

from _paths import rag_root

c = chromadb.PersistentClient(path=str(rag_root() / "index"))
col = c.get_collection("library")
real = col.get(limit=50, include=["documents"])
docs = [d[:1200] for d in real["documents"]]
print(f"{len(docs)} real chunks (truncated ~1200 chars)")

queries = [
    "Principles of psychological warfare",
    "EXPLOITATION of human intelligence sources",
    "close quarter combat techniques",
    "équitation techniques d'entraînement",
    "taekwondo hapkido korean martial arts",
]

models = [
    ("BGE-M3 (baseline)", "BAAI/bge-m3", {}),
    ("multilingual-e5-small", "intfloat/multilingual-e5-small", {}),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", {}),
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", {}),
]

full_chunks = 127856  # measured total corpus chunks

print(f"\n{'model':<26}{'dim':>5}{'c/s':>9}{'full-run hrs':>14}")
for name, mid, kw in models:
    try:
        t = time.time()
        m = SentenceTransformer(mid, device="cpu")
        m.encode(docs[:4], show_progress_bar=False)  # warmup + thread init
        e = time.time() - t
        t0 = time.time()
        embs = m.encode(docs, show_progress_bar=False, convert_to_numpy=True)
        t1 = time.time()
        cps = len(docs) / (t1 - t0)
        hrs = full_chunks / cps / 3600
        tq = time.time()
        m.encode(queries, show_progress_bar=False)
        qs = time.time() - tq
        print(f"{name:<26}{embs.shape[1]:>5}{cps:>9.1f}{hrs:>14.2f}   (load {e:.0f}s, {len(queries)}q {qs:.2f}s)")
    except Exception as ex:
        print(f"{name:<26} FAILED: {ex}")