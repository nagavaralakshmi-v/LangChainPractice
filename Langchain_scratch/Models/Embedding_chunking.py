from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ Create the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2️⃣ Your sample documents
docs = [
    Document(page_content="LangChain is a framework for building LLM-powered apps."),
    Document(page_content="Vector stores help you store and search embeddings."),
    Document(page_content="Embeddings turn text into numerical vectors.")
]

# 3️⃣ Create a FAISS vector store from the docs
vector_store = FAISS.from_documents(docs, embedding=embeddings)

# 4️⃣ Now query it!
query = "How do I store text embeddings?"
results = vector_store.similarity_search(query, k=2)

for idx, doc in enumerate(results):
    print(f"Result {idx+1}: {doc.page_content}")
