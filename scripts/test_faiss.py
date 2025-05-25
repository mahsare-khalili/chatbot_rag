# chatbot_rag/scripts/test_faiss.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Use free SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is great for predictions.",
    "Neural networks are a type of machine learning model.",
    "I love making coffee in the morning."
]

# Generate embeddings
embeddings = model.encode(sentences)

# Initialize Faiss index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Search
query = "What are neural networks?"
query_vec = model.encode([query])
D, I = index.search(np.array(query_vec), k=2)

print("Query:", query)
for i in I[0]:
    print("Match:", sentences[i])
