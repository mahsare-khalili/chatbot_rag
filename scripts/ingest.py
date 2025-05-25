import os
import pdfplumber
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# ---- Config ----
PDF_DIR = "./pdfs"
CHUNK_SIZE = 500  # characters
INDEX_FILE = "./vectorstore/index.faiss"
META_FILE = "./vectorstore/metadata.json"

# Ensure output directory exists
os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

# ---- Load Embedding Model (to converts text into semantic vectors) ----
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Helper: Chunk Text ----
def chunk_text(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]

# ---- Helper: Load PDFs and Extract Text ----
def load_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            with pdfplumber.open(path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"
                documents.append((filename, full_text.strip()))
    return documents

# ---- Main Ingestion ----
def ingest():
    docs = load_pdfs(PDF_DIR)
    embeddings = []
    metadata = []

    for doc_name, content in docs:
        chunks = chunk_text(content, CHUNK_SIZE)
        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk)
            embeddings.append(emb)
            metadata.append({
                "source": doc_name,
                "chunk_id": i,
                "text": chunk
            })

    # Create Faiss index
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))

    # Save index & metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Ingested {len(embeddings)} chunks from {len(docs)} documents.")
    print(f"Index saved to: {INDEX_FILE}")
    print(f"Metadata saved to: {META_FILE}")

if __name__ == "__main__":
    ingest()
