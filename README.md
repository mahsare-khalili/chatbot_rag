# 📚 RAG-Powered PDF Chatbot
Answer questions from your own PDFs using a local LLM (FLAN-T5), semantic search (FAISS), and intelligent fallback.

---

## 🧠 Features

- **Local LLM (FLAN-T5):** Generate answers without relying on external APIs.
- **Semantic Search with FAISS:** Efficiently retrieve relevant document chunks.
- **Intelligent Fallbacks:** Utilize DuckDuckGo and Wikipedia when necessary.
- **Dual Interface:** Interact via both CLI and Streamlit-based UI.
- **Modular Architecture:** Clean separation of concerns for maintainability.

---

## 🗂️ Project Structure

        project/
        ├── main.py                    # Main entry point. Tries vector + RAG, falls back if needed
        ├── configs/
        │   └── settings.py            # Configuration settings for the project
        ├── core/
        │   └── chatbot_core.py        # Shared logic for app & CLI
        ├── ingest/
        │   └── run_ingest.py          # Loads PDFs, chunks text, embeds via SentenceTransformer, saves to FAISS + metadata
        ├── retriever/
        │   └── langchain_retriever.py # Retrieves top-K relevant chunks using vector similarity search
        ├── models/
        │   └── langchain_qa.py        # Generates an answer using Flan-T5 from HuggingFace Transformers
        ├── fallback/
        │   ├── wikipedia_fallback.py  # Fallback using Wikipedia (via wikipedia package)
        │   └── duckduckgo_fallback.py # Fallback using DuckDuckGo API (if enabled in settings)
        ├── utils/
        │   └── chunking.py            # Shared chunking logic (sliding windows, overlap, etc.)
        ├── pdfs/                      # Input directory for raw documents (PDFs)
        ├── vectorstore/
        │   ├── index.faiss            # FAISS vector index file
        │   └── metadata.json          # Metadata for associated chunks (source doc, chunk ID, etc.)
        ├── requirements.txt           # Dependency list for Python environment
        ├── README.md                  # Project overview, usage, structure, and setup instructions
        └── .env                       # (Optional) API keys or environment variables

---

## 🛠️ Setup Instructions

1. **Clone the Repository:**

   ```bash
        git clone https://github.com/yourusername/chatbot_rag.git
        cd chatbot_rag
2. **Create a Virtual Environment:**

   ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install Dependencies:**

   ```bash
        pip install -r requirements.txt
4. **Configure Environment Variables:(Optional)**

    Create a .env file in the root directory and add any necessary environment variables, such as API keys.


5. **Ingest PDFs:**
    Place your PDF files into the pdfs/ directory.

    Run the ingestion script:
   ```bash
        python ingest/run_ingest.py
## 🚀 Usage
## Streamlit UI
- Launch the Streamlit application:
   ```bash
        streamlit run app.py
Navigate to http://localhost:8501 in your browser to interact with the chatbot.

## Command-Line Interface
- Run the CLI application:
   ```bash
        python main.py
You will be prompted to enter your question directly in the terminal.

## 🧱 Architecture Overview
Note: Replace architecture.png with your actual architecture diagram illustrating the data flow between components such as PDF ingestion, vectorstore, retriever, LLM, and fallback mechanisms.

## 💬 Sample Q&A Transcript
User: What is the main topic of the document "AI_Research.pdf"?

Chatbot: The document "AI_Research.pdf" primarily discusses advancements in artificial intelligence, focusing on machine learning algorithms and their applications in data analysis.

Source: AI_Research.pdf, chunk 3

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) – Document retrieval and QA framework
- [HuggingFace Transformers](https://github.com/huggingface/transformers) – Pretrained LLMs
- [FAISS](https://github.com/facebookresearch/faiss) – Fast similarity search
- [Streamlit](https://github.com/streamlit/streamlit) – Web UI

## 🔖 Versioning
This project uses semantic versioning. The current version is v0.1.

## 📧 Contact

Questions or suggestions? Reach out via [mahsare.khalili@gmail.com](mailto:mahsare.khalili@gmail.com) or visit [https://github.com/mahsare-khalili](https://github.com/mahsare-khalili).
