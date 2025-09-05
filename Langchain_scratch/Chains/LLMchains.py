import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()

# Define prompt
template = "Translate the following sentence into French: {sentence}"
prompt = PromptTemplate.from_template(template)

# Define Gemini model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

# LCEL way: chain prompt → llm
chain = prompt | llm

# Run chain with invoke (instead of deprecated .run)
result = chain.invoke({"sentence": "I love programming with LangChain"})
print("🔹 Output:", result.content)
