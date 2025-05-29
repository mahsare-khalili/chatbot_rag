import streamlit as st
from core.chatbot_core import run_query

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
                answer, sources, score = run_query(question)

            st.subheader("Answer")
            st.write(answer)

            if sources:
                st.subheader("Source")
                doc = sources[0]
                st.write(f"- **{doc.metadata['source']}**, chunk {doc.metadata['chunk_id']}")
                st.write(f"🔍 Cosine similarity: {score:.3f}")
            else:
                st.info("No sources (fallback response).")
