from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Sample large text
text = """
LangChain makes it easy to build applications powered by LLMs.
It helps with prompt management, chaining, retrieval, and more.
Text splitters break large documents into smaller chunks.
This makes it easier to embed and search efficiently.
FAISS is a fast vector store for similarity search.
"""

# Step 1: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_text(text)

print("Chunks:")
for i, c in enumerate(chunks):
    print(f"Chunk {i+1}: {c}")

# Step 2: Convert chunks into embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Step 3: Store embeddings in FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)

# Step 4: Query (user asks something)
query = "What does LangChain help with?"
docs = vectorstore.similarity_search(query)

print("\nRetrieved Documents:")
for doc in docs:
    print(doc.page_content)
