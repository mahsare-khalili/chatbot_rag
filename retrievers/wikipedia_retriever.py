# retrievers/wikipedia_retriever.py
import wikipedia
from .base_retriever import BaseRetriever

class WikipediaRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 3):
        try:
            summaries = wikipedia.summary(query, sentences=3)
            return [{"source": "Wikipedia", "chunk_id": 0, "text": summaries}]
        except Exception:
            return []
