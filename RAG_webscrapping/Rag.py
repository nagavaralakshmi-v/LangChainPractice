import asyncio
import sys
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Fix for Streamlit + gRPC AsyncIO ---
if sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

# --- Load Gemini API key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.title("🌐 Gemini RAG Chatbot (Website Only)")

# Input website URL
url = st.text_input(
    "Enter website URL:", "https://python.langchain.com/docs/get_started/introduction"
)

if url:
    # 1️⃣ Load website content
    loader = WebBaseLoader(url)
    documents = loader.load()

    if documents:
        # 2️⃣ Create embeddings & vector store (sync mode)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key, async_client=False
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        # 3️⃣ Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3
        )

        # 4️⃣ RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True
        )

        # 5️⃣ Ask question
        user_question = st.text_input("Ask a question about the website:")
        if user_question:
            result = qa_chain.invoke({"query": user_question})

            st.markdown("### Answer:")
            st.write(result["result"])

            st.markdown("### Source(s):")
            for doc in result["source_documents"]:
                st.write("-", doc.metadata.get("source", "Unknown"))
