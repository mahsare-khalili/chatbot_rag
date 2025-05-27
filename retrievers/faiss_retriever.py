# retrievers/faiss_retriever.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .base_retriever import BaseRetriever

class FAISSRetriever(BaseRetriever):
    def __init__(self, index_file: str, meta_file: str):
        self.index = faiss.read_index(index_file)
        with open(meta_file, "r") as f:
            self.metadata = json.load(f)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(query_vec, top_k)
        results = []
        for i in I[0]:
            if i < len(self.metadata):
                results.append(self.metadata[i])
        return results
