# configs/settings.py

# Where your PDFs live
PDF_DIR = "./pdfs"

# Where to write FAISS index + metadata
INDEX_DIR = "./vectorstore"

# Embedding model name (used by SentenceTransformerEmbeddings)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM model for answer generation
LLM_MODEL = "google/flan-t5-base"

# Cosine similarity threshold (0.0–1.0).
# If best_score < this, we fallback.
DISTANCE_THRESHOLD = 0.40

# Turn DuckDuckGo fallback on/off
ENABLE_DUCKDUCKGO = True
