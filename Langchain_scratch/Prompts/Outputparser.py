from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnableLambda

# 1️⃣ Few-shot examples
examples = [
    {"english": "I love programming", "french": "J'adore programmer"},
    {"english": "She goes to school", "french": "Elle va à l'école"}
]

# 2️⃣ Template for each example
example_prompt = PromptTemplate(
    input_variables=["english", "french"],
    template="English: {english}\nFrench: {french}"
)

# 3️⃣ Few-shot prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="English: {user_input}\nFrench:",
    input_variables=["user_input"],
)

# 4️⃣ Initialize Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# 5️⃣ Custom parser function
def parse_translation(response):
    # Gemini returns ChatResult object; get content
    text = response.content if hasattr(response, "content") else str(response)
    if ":" in text:
        text = text.split(":", 1)[-1]
    return text.strip()

parser_runnable = RunnableLambda(parse_translation)

# 6️⃣ Build LCEL pipeline
pipeline = few_shot_prompt | llm | parser_runnable

# 7️⃣ Run pipeline
query = "I am learning LangChain"
result = pipeline.invoke({"user_input": query})

# 8️⃣ Print output
print("English Query:", query)
print("French Translation:", result)
