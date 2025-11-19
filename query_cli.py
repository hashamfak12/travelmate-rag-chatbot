#!/usr/bin/env python
import argparse, faiss, numpy as np, time
from travelmate.embeddings import load_embedder, embed_texts
from travelmate.index_store import IndexStore
from travelmate.generator import generate_answer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    store = IndexStore(index_dir=args.index_dir)
    index, metadatas, texts = store.load()

    model = load_embedder()
    s = time.time()
    q_emb = embed_texts(model, [args.question]).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, args.k)
    elapsed = (time.time() - s) * 1000.0

    retrieved = []
    for rank, idx in enumerate(I[0]):
        if idx == -1: continue
        meta = metadatas[idx]
        retrieved.append({"rank": rank+1, "text": texts[idx], "source": meta["source"], "chunk_id": meta["chunk_id"]})

    answer = generate_answer(args.question, retrieved)
    print("\n=== Answer ===\n")
    print(answer)
    print(f"\nRetrieval latency: {elapsed:.1f} ms")
    print("\n=== Top citations ===\n")
    for r in retrieved:
        print(f"[{r['rank']}] {r['source']} (chunk {r['chunk_id']})\n{r['text'][:200]}...\n")

if __name__ == "__main__":
    main()
