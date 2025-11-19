import os, json, faiss, numpy as np

class IndexStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "metadatas.json")
        self.texts_path = os.path.join(index_dir, "texts.json")

    def save(self, index, metadatas, texts):
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadatas, fh, ensure_ascii=False, indent=2)
        with open(self.texts_path, "w", encoding="utf-8") as fh:
            json.dump(texts, fh, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Missing index at {self.index_path}")
        index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as fh:
            metadatas = json.load(fh)
        with open(self.texts_path, "r", encoding="utf-8") as fh:
            texts = json.load(fh)
        return index, metadatas, texts
