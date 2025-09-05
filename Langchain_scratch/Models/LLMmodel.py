from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

# Gemini text model
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest")

response = llm.invoke("Write a story about the ocean.")
print(response)
