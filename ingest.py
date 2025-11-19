#!/usr/bin/env python
import os, argparse, json, uuid, faiss, numpy as np
from travelmate.chunking import chunk_texts
from travelmate.embeddings import load_embedder, embed_texts
from travelmate.index_store import IndexStore

def load_text_files(data_dir):
    docs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".txt", ".md")):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                docs.append({"doc_id": str(uuid.uuid4()), "source": path, "text": text})
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--chunk_overlap", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    docs = load_text_files(args.data_dir)
    if not docs:
        print("No .txt or .md files found:", args.data_dir); return

    all_chunks, metadatas = [], []
    for d in docs:
        chunks = chunk_texts(d["text"], chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadatas.append({"doc_id": d["doc_id"], "source": d["source"], "chunk_id": i})

    model = load_embedder()
    embeddings = embed_texts(model, all_chunks).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    store = IndexStore(index_dir=args.index_dir)
    store.save(index=index, metadatas=metadatas, texts=all_chunks)
    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents.")

if __name__ == "__main__":
    main()
