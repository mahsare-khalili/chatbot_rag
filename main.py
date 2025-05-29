from core.chatbot_core import run_query

if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    if not question:
        print("Please enter a non-empty question.")
        exit(1)

    answer, sources, score = run_query(question)

    print("\n=== Answer ===")
    print(answer)

    if sources:
        print("\n=== Source ===")
        doc = sources[0]
        md = doc.metadata
        print(f"- {md['source']}, chunk {md['chunk_id']} (sim={score:.3f})")
    else:
        print("\n[Info] Fallback source (no document match).")
