# chatbot_rag/models/langchain_qa.py

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

from retriever.langchain_retriever import retriever
from configs.settings import LLM_MODEL

# Initialize a Hugging Face pipeline for text-generation
generation_pipeline = pipeline(
    "text2text-generation",
    model=LLM_MODEL,
    tokenizer=LLM_MODEL,
    max_length=200,
    do_sample=True,
    temperature=0.3,
    top_k=10,
    top_p=0.95
)

llm = HuggingFacePipeline(pipeline=generation_pipeline)

# Build a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

def answer_with_chain(query: str):
    """
    Returns:
      {
        "result": <generated answer string>,
        "source_documents": [Document, ...]
      }
    """
    return qa_chain.invoke({"query": query})