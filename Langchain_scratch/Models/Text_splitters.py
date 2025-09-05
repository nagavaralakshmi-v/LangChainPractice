from langchain.text_splitter import RecursiveCharacterTextSplitter

# A large text (e.g., from a book or article)
text = """
LangChain makes it easy to build applications powered by LLMs.
It helps with prompt management, chaining, retrieval, and more.
Text splitters break large documents into smaller chunks.
This makes it easier to embed and search efficiently.
"""

# Initialize a splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,    # max characters in a chunk
    chunk_overlap=10  # overlap between chunks (helps context flow)
)

# Split text
chunks = text_splitter.split_text(text)

# Print results
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
