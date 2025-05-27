# main.py (token-based presence check)

import re
from retriever.langchain_retriever import get_relevant_chunks_with_scores
from models.langchain_qa import answer_with_chain
from fallback.duckduckgo_fallback import query_duckduckgo
from fallback.wikipedia_fallback import query_wikipedia
from configs.settings import ENABLE_DUCKDUCKGO, DISTANCE_THRESHOLD

def extract_query_tokens(query: str) -> list[str]:
    """
    Strip punctuation, lowercase, and split into words.
    Returns only tokens longer than 2 characters (to skip 'is', 'what', etc.).
    """
    # Remove non-alphanumeric characters
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', " ", query.lower())
    tokens = cleaned.split()
    # Keep only tokens that are likely to carry content
    return [t for t in tokens if len(t) > 2]

if __name__ == "__main__":
    q = input("Enter your question: ")

    #  Retrieve top-k chunks + cosine scores
    docs_and_scores = get_relevant_chunks_with_scores(q, k=3)
    docs, scores = zip(*docs_and_scores) if docs_and_scores else ([], [])

    # TOKEN PRESENCE CHECK
    tokens = extract_query_tokens(q)  # e.g. ["what", "coffee"]
    text = " ".join(d.page_content.lower() for d in docs)
    # Check if any *meaningful* token appears
    if docs and not any(tok in text for tok in tokens if tok not in ("what", "how", "the", "is")):
        print(f"[Token check] None of {tokens} appear in the context → forcing fallback.")
        docs, scores = [], []

    # Threshold logic
    best_score = scores[0] if scores else float("-inf")
    print(f"[Debug] Best cosine similarity: {best_score:.3f}")

    if not docs or best_score < DISTANCE_THRESHOLD:
        print(f"No good match (best {best_score:.3f} < threshold {DISTANCE_THRESHOLD:.3f}).")
        if ENABLE_DUCKDUCKGO:
            print("\n=== DuckDuckGo fallback ===")
            print(query_duckduckgo(q))
        else:
            print("\n=== Wikipedia fallback ===")
            print(query_wikipedia(q))
    else:
        # Good match → run RAG QA
        res = answer_with_chain(q)
        print("\n=== Answer ===\n" + res["result"])
        print("\n=== Sources ===")
        for doc in res["source_documents"]:
            md = doc.metadata
            print(f"- {md['source']}, chunk {md['chunk_id']} (sim={best_score:.3f})")
