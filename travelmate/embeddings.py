from sentence_transformers import SentenceTransformer

def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def embed_texts(model, texts):
    return model.encode(texts, normalize_embeddings=False, convert_to_numpy=True)
