import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
 
# Load .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
# Initialize Gemini as LLM for LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
 
# ---- Define Tools ----
# 1. Search tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Useful for searching the web for current information."
)
 
# 2. Calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
 
# List of tools
tools = [search_tool, calculator]
 
# Create agent with tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)
 
# ---- Streamlit UI ----
st.title("🤖 Gemini Agent Chatbot with Tools")
 
user_input = st.text_input("Ask me something:")
 
if user_input:
    response = agent.run(user_input)
    st.write("### Bot:")
    st.write(response)