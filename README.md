# TravelMate — RAG Chatbot for Travel Guides (Full Starter Kit)

A **ready-to-run** Retrieval-Augmented Generation (RAG) project aligned with your proposal.
It includes: FAISS retrieval, Streamlit UI, optional LLM generation, and a tiny **evaluation harness**.

![Architecture](assets/architecture.png)

## ✨ What's included
- **RAG pipeline**: chunk → embed → FAISS index → retrieve → generate (with citations)
- **Streamlit UI**: ask questions, view sources, adjust top-k
- **Eval harness**: `evaluate.py` + `eval/questions.jsonl` for precision@k and latency
- **Sample data**: Paris, Tokyo, London, Rome, Bangkok, New York (replace with Wikivoyage excerpts later)
- **OpenAI (optional)**: set `OPENAI_API_KEY` to enable LLM answers

---

## 🔧 Quick Start

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Build the FAISS index from /data
python ingest.py --data_dir ./data --index_dir ./index --chunk_size 500 --chunk_overlap 50

# Try a CLI question
python query_cli.py --index_dir ./index --question "How to get from CDG to central Paris?"

# Run the Streamlit app
streamlit run app_streamlit.py
```

### Optional: enable LLM answers
```bash
cp .env.example .env
# add your OPENAI_API_KEY
```

---

## 🧪 Evaluation (precision@k + latency)

Edit `eval/questions.jsonl` to add your own QA pairs. Then run:
```bash
python evaluate.py --index_dir ./index --k 5
```

You'll get a small report with:
- **Precision@k** — fraction of queries where at least one retrieved passage contains an expected **keyword** (simple heuristic)
- **Latency** — average retrieval time (ms)

> Note: This is a lightweight academic evaluation for check-ins. For the final project, expand to more queries and better metrics.

---

## 🧱 Project Layout

```
travelmate_project_full/
  assets/architecture.png
  data/
    paris.txt, tokyo.txt, london.txt, rome.txt, bangkok.txt, newyork.txt
  eval/questions.jsonl
  ingest.py
  query_cli.py
  evaluate.py
  app_streamlit.py
  travelmate/
    __init__.py
    chunking.py
    embeddings.py
    index_store.py
    generator.py
    utils.py
  requirements.txt
  .env.example
  README.md
```

---

## 🗺️ Roadmap
1) Replace sample /data with curated, cleaned text from Wikivoyage & official guides.
2) Expand `eval/questions.jsonl` to 25–50 questions.
3) Improve prompts and try stronger embedding models.
4) Add HTML cleaning & metadata tagging (city/section) in `utils.py`.
5) (Optional) UI polish: city filter, file uploader, live re-indexing.
