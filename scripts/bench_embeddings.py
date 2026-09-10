#!/usr/bin/env python3
"""bench_embeddings.py - Compare BGE-M3 vs Qwen3-Embedding-0.6B speed & quality on local corpus."""
import time
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.set_num_threads(16)
from sentence_transformers import SentenceTransformer

EN = 200
texts = [
    f"Document chunk number {i} discussing retrieval augmented generation, vector databases, "
    f"and knowledge management for local corpora across multiple languages including French"
    f" Spanish Korean German and content classification bibliographique diversifié."
    for i in range(EN)
]
queries = [
    "What are the key principles of psychological warfare?",
    "How does mind control work according to research?",
    "Effective techniques for close quarter combat",
    "Méthodes de dressage équin efficace",
    "TKDE hapkido techniques korean martial arts",
]

# Extract a real chunk sample from the index to use as realistic docs
try:
    import chromadb
    from pathlib import Path
    from _paths import rag_root
    c = chromadb.PersistentClient(path=str(rag_root() / "index"))
    col = c.get_collection("library")
    real = col.get(limit=50, include=["documents"])
    if real["documents"]:
        docs = [d[:600] for d in real["documents"]]
        texts = docs
        EN = len(texts)
        print(f"Using {EN} real indexed chunks for doc-side benchmark")
    else:
        print(f"Using {EN} synthetic chunks")
except Exception as e:
    print(f"Synthetic chunks ({e})")

def bench(model_name, model_kwargs=None, tag="document"):
    t_load = time.time()
    m = SentenceTransformer(model_name, device="cpu", **(model_kwargs or {}))
    load_s = time.time() - t_load
    # warmup (also pulls in first-encode overhead/thread init)
    m.encode(texts[:4], show_progress_bar=False)
    t0 = time.time()
    embs = m.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    t1 = time.time()
    n = len(texts)
    cps = n / (t1 - t0)
    print(f"\n=== {model_name} ===")
    print(f"  load: {load_s:.1f}s   dim: {embs.shape[1]}   threads=16")
    print(f"  {n} chunks in {t1-t0:.1f}s  = {cps:.1f} chunks/sec")
    # query timing
    t0 = time.time()
    m.encode(queries, show_progress_bar=False, convert_to_numpy=True)
    t1 = time.time()
    print(f"  {len(queries)} queries in {t1-t0:.2f}s")
    return m, embs

print("BGE-M3 vs Qwen3-Embedding-0.6B benchmark (CPU, 16 threads)")
print("=" * 60)

# BGE-M3
m1, e1 = bench("BAAI/bge-m3")

# Qwen3-Embedding-0.6B
m2, e2 = bench("Qwen/Qwen3-Embedding-0.6B", {"trust_remote_code": True})

# Quick quality sanity: can each model retrieve the most similar chunk to each query?
print("\n--- retrieval@1 sanity (self-similarity check on real docs) ---")
emb1 = m1.encode(docs[:20], show_progress_bar=False)
emb2 = m2.encode(docs[:20], show_progress_bar=False)
import numpy as np
def top1(embs, idx):
    sim = (embs @ embs[idx]) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(embs[idx]))
    return int(np.argsort(sim)[-2])  # exclude self
for name, embs in [("BGE-M3", emb1), ("Qwen3-0.6B", emb2)]:
    found = sum(1 for i in range(0, 20) if top1(embs, i) in range(18, 20))
    print(f"  {name}: {found}/20 chunks find a near-neighbor via nearest chunk adjacency")