# chatbot_rag/ingest/run_ingest.py
import os, pdfplumber, json
import numpy as np
import faiss

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore

from configs.settings import PDF_DIR, INDEX_DIR, EMBEDDING_MODEL
from utils.chunking import chunk_text

# Ensure vectorstore directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# Load & chunk PDFs into LangChain Documents
docs: list[Document] = []
for file in os.listdir(PDF_DIR):
    if not file.endswith('.pdf'):
        continue
    path = os.path.join(PDF_DIR, file)
    with pdfplumber.open(path) as pdf:
        full_text = ''.join(page.extract_text() or '' for page in pdf.pages)
    for idx, chunk in enumerate(chunk_text(full_text, size=750), start=1):
        docs.append(Document(
            page_content=chunk, 
            metadata={'source': file, 'chunk_id': idx}))

# Embed all chunks
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
# embed_documents returns a list of vectors
vectors = embeddings.embed_documents([d.page_content for d in docs])

# Convert to NumPy & normalize to unit length
xb = np.array(vectors, dtype="float32")
faiss.normalize_L2(xb)

# Build an IndexFlatIP (inner-product) index
dim = xb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(xb)

# We need a mapping from FAISS row → Document
#    FAISS.load_local expects index_to_docstore_id + an in-memory docstore
index_to_docstore_id = {i: str(i) for i in range(len(docs))}
docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})

# Wrap in LangChain FAISS and save
vectorstore = FAISS(embedding_function=embeddings, 
                    index=index, 
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id)
# Save locally for LangChain load
vectorstore.save_local(INDEX_DIR)

# Save raw metadata
with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
    json.dump([d.metadata for d in docs], f, indent=2)

print(f"Saved vectorstore ({len(docs)} docs) to {INDEX_DIR}")
print(f"Ingested {len(docs)} chunks.")