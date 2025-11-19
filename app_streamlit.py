import os, faiss, numpy as np, streamlit as st
from travelmate.embeddings import load_embedder, embed_texts
from travelmate.index_store import IndexStore
from travelmate.generator import generate_answer

st.set_page_config(page_title="TravelMate", page_icon="🧳", layout="wide")

with st.sidebar:
    st.header("TravelMate")
    st.write("RAG chatbot for travel guides. Build index with `ingest.py` first.")
    index_dir = st.text_input("Index directory", "./index")
    k = st.slider("Top-k passages", 1, 10, 5)
    if st.button("Load index"):
        try:
            store = IndexStore(index_dir=index_dir)
            index, metadatas, texts = store.load()
            st.session_state["index"] = index
            st.session_state["metadatas"] = metadatas
            st.session_state["texts"] = texts
            st.success(f"Loaded index with {len(texts)} chunks.")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

st.title("🧳 TravelMate — RAG Chatbot")
question = st.text_input("Ask a travel question (e.g., 'best way from CDG to Paris center?')")
if st.button("Ask") and question:
    if "index" not in st.session_state:
        st.warning("Load the index in the sidebar first.")
    else:
        embedder = load_embedder()
        q_emb = embed_texts(embedder, [question]).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = st.session_state["index"].search(q_emb, k)

        retrieved = []
        for rank, idx in enumerate(I[0]):
            if idx == -1: continue
            meta = st.session_state["metadatas"][idx]
            retrieved.append({"rank": rank+1, "text": st.session_state["texts"][idx], "source": meta["source"], "chunk_id": meta["chunk_id"]})

        answer = generate_answer(question, retrieved)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Citations")
        for r in retrieved:
            with st.expander(f"[{r['rank']}] {r['source']} (chunk {r['chunk_id']})"):
                st.write(r["text"])
