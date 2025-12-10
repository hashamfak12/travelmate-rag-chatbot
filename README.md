# TravelMate – RAG Chatbot for Travel Guides

TravelMate is a Retrieval-Augmented Generation (RAG) chatbot that I built for my CS 156 Final Project. The goal of the project is to answer travel-related questions using actual travel guides instead of relying only on an LLM. The system retrieves information from my small dataset of city notes (Paris, Tokyo, London, Rome, Bangkok, and New York) and uses those snippets to answer user questions.
I used FAISS for indexing, SentenceTransformer embeddings for encoding text, and Streamlit for the user interface.

---

## What the Project Can Do

- Search through real travel guide files and pull out relevant information  
- Show the top-k retrieved snippets along with citations  
- Provide answers through a clean Streamlit interface  
- Optionally support LLM-generated answers (if an API key is added)  
- Evaluate retrieval quality using precision@k and latency  

Even without an LLM enabled, the system still works and shows grounded answers through retrieved snippets.

---

## Project Structure

data/ — Travel guide text files (6 cities)  
index/ — FAISS index created after running ingest.py  
eval/questions.json — Questions used to evaluate precision@k  
app_streamlit.py — Streamlit UI  
ingest.py — Builds FAISS index  
evaluate.py — Evaluation script  
embeddings.py — Embedding functions  
index_store.py — FAISS index loader/search wrapper  
generator.py — Generates answers (fallback mode if no LLM)  

---
## How to Run the Project

### 1. Create and activate a virtual environment
python -m venv .venv  
.venv\Scripts\activate   

### 2. Install Requirements
pip install -r requirements.txt  

### 3. Build the FAISS index
python ingest.py --data_dir ./data --index_dir ./index --chunk_size 500 --chunk_overlap 50  

### 4. Run the Streamlit app
streamlit run app_streamlit.py




