# Rag_with_Gemini.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Step 1: Sample documents
# -------------------------------
texts = [
    "LangChain makes it easy to build applications powered by LLMs.",
    "It helps with prompt management, chaining, retrieval, and more.",
    "Text splitters break large documents into smaller chunks.",
    "This makes it easier to embed and search efficiently.",
    "FAISS is a fast vector store for similarity search."
]

# Convert to Document objects
docs = [Document(page_content=t) for t in texts]

# -------------------------------
# Step 2: Split text into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
all_chunks = text_splitter.split_documents(docs)

# -------------------------------
# Step 3: Create embeddings
# -------------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="learnlm-2.0-flash",  # Replace with a supported embedding model
    api_key=api_key
)

# -------------------------------
# Step 4: Create FAISS vector store
# -------------------------------
vectorstore = FAISS.from_documents(all_chunks, embedding_model)

# -------------------------------
# Step 5: Setup RetrievalQA chain
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  # Replace with a supported chat model
    api_key=api_key
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # retrieve top 2 similar chunks
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",   # you can use 'map_reduce', 'stuff', etc.
    retriever=retriever
)

# -------------------------------
# Step 6: Ask questions
# -------------------------------
query = "What is FAISS used for?"
answer = qa_chain.run(query)

print("Question:", query)
print("Answer:", answer)
