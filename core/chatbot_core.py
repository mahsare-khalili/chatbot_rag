import re
from typing import List, Tuple

from retriever.langchain_retriever import get_relevant_chunks_with_scores
from fallback.duckduckgo_fallback import query_duckduckgo
from fallback.wikipedia_fallback import query_wikipedia
from transformers import pipeline
from configs.settings import ENABLE_DUCKDUCKGO, DISTANCE_THRESHOLD, LLM_MODEL
from langchain.docstore.document import Document

# --- HF pipeline (same as before) ---
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

STOP_WORDS = {
    "of", "the", "is", "in", "to", "and", "for", "on", "with", "as", "by",
    "at", "from", "or", "be", "this", "that", "it", "its", "no", "not", "but",
    "we", "you", "are", "was", "were", "so", "if", "my", "can", "has", "have"
}

def extract_query_tokens(query: str) -> List[str]:
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

def run_query(question: str) -> Tuple[str, List[Document], float]:
    docs_and_scores = get_relevant_chunks_with_scores(question, k=3)
    if not docs_and_scores:
        return (
            (ENABLE_DUCKDUCKGO and query_duckduckgo(question))
            or query_wikipedia(question),
            [],
            -1.0
        )

    tokens = extract_query_tokens(question)
    meaningful = [t for t in tokens if t not in ("what", "how", "the", "is")]

    filtered = [
        (doc, score)
        for doc, score in docs_and_scores
        if any(tok in doc.page_content.lower() for tok in meaningful)
    ]
    if not filtered:
        return (
            (ENABLE_DUCKDUCKGO and query_duckduckgo(question))
            or query_wikipedia(question),
            [],
            -1.0
        )

    best_doc, best_score = filtered[0]

    has_token_match = any(tok in best_doc.page_content.lower() for tok in meaningful)
    if not has_token_match and best_score < DISTANCE_THRESHOLD:
        return (
            (ENABLE_DUCKDUCKGO and query_duckduckgo(question))
            or query_wikipedia(question),
            [],
            best_score
        )

    answer = generate_answer_from_chunk(best_doc.page_content, question)
    return answer, [best_doc], best_score
