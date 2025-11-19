#!/usr/bin/env python
import argparse, json, time, numpy as np, faiss
from typing import List, Dict
from travelmate.embeddings import load_embedder, embed_texts
from travelmate.index_store import IndexStore

def load_eval(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def keyword_hit(text: str, keywords):
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--eval_file", default="./eval/questions.jsonl")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    eval_items = load_eval(args.eval_file)
    store = IndexStore(index_dir=args.index_dir)
    index, metadatas, texts = store.load()
    embedder = load_embedder()

    hits = 0
    total = 0
    latencies = []

    for item in eval_items:
        q = item["question"]
        exp_keywords = item.get("expected_keywords", [])
        s = time.time()
        q_emb = embed_texts(embedder, [q]).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, args.k)
        latencies.append((time.time() - s) * 1000.0)

        found = False
        for idx in I[0]:
            if idx == -1: continue
            if keyword_hit(texts[idx], exp_keywords):
                found = True
                break
        hits += 1 if found else 0
        total += 1

    prec_at_k = hits / max(1, total)
    avg_latency = sum(latencies) / max(1, len(latencies))
    print(f"Queries: {total}")
    print(f"Precision@{args.k}: {prec_at_k:.2f}")
    print(f"Avg retrieval latency: {avg_latency:.1f} ms")

if __name__ == "__main__":
    main()
