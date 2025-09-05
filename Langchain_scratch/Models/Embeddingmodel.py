from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# 1️⃣ Create the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key="YOUR_API_KEY")

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

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = embed.embed_query("LangChain is amazing!")
print(vector[:5])  # first 5 numbers of the vector
