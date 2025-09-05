import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

response = chat_model.invoke([HumanMessage(content="Tell me a joke about cats.")])
print(response.content)



# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage, SystemMessage

# # Load Gemini chat model
# chat = ChatGoogleGenerativeAI(model="gemini-pro", api_key="YOUR_API_KEY")

# # Create structured messages
# messages = [
#     SystemMessage(content="You are a helpful tutor who explains in simple terms."),
#     HumanMessage(content="Explain LangChain Chat Models in one line.")
# ]

# # Call the model
# response = chat.invoke(messages)

# print(response.content)
