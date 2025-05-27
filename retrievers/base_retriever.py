# retrievers/base_retriever.py
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3):
        pass
