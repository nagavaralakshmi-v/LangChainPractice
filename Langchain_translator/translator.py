import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
 
# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
 
# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.3
)
 
# Define translation prompt
prompt = ChatPromptTemplate.from_template(
    "Translate the following text into {language}:\n\n{text}"
)
 
st.title("🌍 Language Translator Chatbot")
#st.write("Powered by LangChain + Gemini Flash 1.5")
 
# Sidebar for selecting target language
language = st.sidebar.selectbox(
    "Select Target Language",
    ["French", "Spanish", "German", "Hindi", "Telugu", "Chinese", "Japanese"]
)
 
# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
# User input
if user_input := st.chat_input("Enter text to translate..."):
    # Show user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    # Build chain
    chain = prompt | llm
    response = chain.invoke({"language": language, "text": user_input})
 
    # ✅ FIX: Extract the response properly
    if hasattr(response, "content") and isinstance(response.content, list):
        output = response.content[0].text
    elif hasattr(response, "content") and isinstance(response.content, str):
        output = response.content
    else:
        output = str(response)
 
    # Show AI output
    st.session_state.messages.append({"role": "assistant", "content": output})
    with st.chat_message("assistant"):
        st.markdown(output)