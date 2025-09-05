from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

# ✅ Make sure you set your Google API Key
load_dotenv()

# 1️⃣ Load Gemini Chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# 2️⃣ Setup Memory (stores past conversation)
memory = ConversationBufferMemory(memory_key="history", input_key="input")

# 3️⃣ Create Conversation Chain
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# 4️⃣ Try chatting
print(conversation.run("Hello, my name is Navya."))
print(conversation.run("What is my name?"))  # ✅ Should remember your name
print(conversation.run("Can you tell me what we have talked about so far?"))
