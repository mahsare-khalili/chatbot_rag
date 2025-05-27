# chatbot_rag/retriever/langchain_retriever.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

from configs.settings import INDEX_DIR, EMBEDDING_MODEL

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# Load LangChain vectorstore (must match ingest.save_local path)
vectorstore = FAISS.load_local(
    INDEX_DIR, 
    embeddings, 
    allow_dangerous_deserialization=True)

# Expose a LangChain Retriever for QA chains
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def get_relevant_chunks_with_scores(query: str, k: int = 3):
    """
        Returns list of (Document, score) where score is cosine similarity ∈ [–1,1].
    """
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)
