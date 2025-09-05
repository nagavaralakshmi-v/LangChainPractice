import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Prompts
summary_prompt = PromptTemplate.from_template(
    "Summarize the following text in one sentence:\n{text}"
)
translate_prompt = PromptTemplate.from_template(
    "Translate this into French:\n{summary}"
)

# Wrap prompts with LLM into callables
def summarize(inputs):
    return llm.invoke(summary_prompt.format(**inputs))

def translate(summary_text):
    return llm.invoke(translate_prompt.format(summary=summary_text))

# RunnableSequence
chain = RunnableSequence([summarize, translate])

# Input
input_text = {"text": "LangChain makes it easy to build applications powered by LLMs. "
                       "It helps with prompt management, chaining, retrieval, and more."}

# Run
summary_result = chain.steps[0](input_text)
translation_result = chain.steps[1](summary_result)

print("🔹 Summary:", summary_result)
print("🔹 Translation:", translation_result)
