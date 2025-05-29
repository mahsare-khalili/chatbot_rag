# app.py

import streamlit as st
import re

from retriever.langchain_retriever import get_relevant_chunks_with_scores
from fallback.duckduckgo_fallback import query_duckduckgo
from fallback.wikipedia_fallback import query_wikipedia
from transformers import pipeline
from configs.settings import (
    ENABLE_DUCKDUCKGO,
    DISTANCE_THRESHOLD,
    LLM_MODEL,
)

# — HuggingFace generation pipeline —
generation_pipeline = pipeline(
    "text2text-generation",
    model=LLM_MODEL,
    tokenizer=LLM_MODEL,
    max_length=200,
    do_sample=True,
    temperature=0.3,
    top_k=10,
    top_p=0.95,
)


# A minimal set—you can expand as you like.
STOP_WORDS = {
    "of","the","is","in","to","and","for","on","with","as","by","at","from",
    "or","be","this","that","it","its","no","not","but","we","you","are","was",
    "were","so","if","my","can","has","have"
}

def extract_query_tokens(query: str) -> list[str]:
    """
    1) Lowercase + drop punctuation
    2) Split into tokens
    3) Keep tokens length>=2 AND not in STOP_WORDS
    """
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', " ", query.lower())
    tokens = cleaned.split()
    return [t for t in tokens if len(t) >= 2 and t not in STOP_WORDS]

def generate_answer_from_chunk(chunk_text: str, question: str) -> str:
    prompt = (
        "You are a helpful assistant. Use the context below to answer the question.\n\n"
        f"Context:\n{chunk_text.strip()}\n\n"
        f"Question: {question}\nAnswer:"
    )
    out = generation_pipeline(prompt, return_text=True)[0]["generated_text"]
    return out.replace(prompt, "").strip()

def run_query(q: str):
    # 1) Retrieve top-3 chunks + scores
    docs_and_scores = get_relevant_chunks_with_scores(q, k=3)
    if not docs_and_scores:
        # no vector matches → fallback
        return (
            ENABLE_DUCKDUCKGO and query_duckduckgo(q)
        ) or query_wikipedia(q), []

    # 2) Extract and filter by keyword tokens
    tokens = extract_query_tokens(q)
    meaningful = [t for t in tokens if t not in ("what", "how", "the", "is")]

    filtered = [
        (doc, score)
        for doc, score in docs_and_scores
        if any(tok in doc.page_content.lower() for tok in meaningful)
    ]
    if not filtered:
        # no explicit token hits → fallback
        return (
            ENABLE_DUCKDUCKGO and query_duckduckgo(q)
        ) or query_wikipedia(q), []

    # 3) Keep only the single best chunk
    best_doc, best_score = filtered[0]

    # Debug log of similarity
    st.write(f"🧐 [Debug] Best similarity = {best_score:.3f}")

    # 4) Conditional threshold bypass:
    #    If we had an exact token match, trust that chunk regardless of similarity
    has_token_match = any(
        tok in best_doc.page_content.lower() for tok in meaningful
    )

    # 5) Similarity threshold guard (unless we bypass)
    if not has_token_match and best_score < DISTANCE_THRESHOLD:
        return (
            ENABLE_DUCKDUCKGO and query_duckduckgo(q)
        ) or query_wikipedia(q), []

    # 6) RAG answer from that chunk
    answer = generate_answer_from_chunk(best_doc.page_content, q)
    return answer, [best_doc]

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG-Powered Chatbot")
st.write("Ask anything about the ingested PDFs.")

with st.form("query_form"):
    question = st.text_input("Your question:")
    submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer, sources = run_query(question)

            st.subheader("Answer")
            st.write(answer)

            if sources:
                st.subheader("Source")
                doc = sources[0]
                st.write(f"- **{doc.metadata['source']}**, chunk {doc.metadata['chunk_id']}")
            else:
                st.info("No sources (fallback response).")
