from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# 1. Define the examples
examples = [
    {"incorrect": "She go to school every day", "correct": "She goes to school every day"},
    {"incorrect": "He like play football", "correct": "He likes playing football"},
]

# 2. Define how each example should look (example prompt template)
example_prompt = PromptTemplate(
    input_variables=["incorrect", "correct"],
    template="Incorrect: {incorrect}\nCorrect: {correct}"
)

# 3. Create the few-shot prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Incorrect: {user_input}\nCorrect:",   # where user input goes
    input_variables=["user_input"],
)

# 4. Format the prompt with a new query
final_prompt = few_shot_prompt.format(user_input="She eat pizza yesterday")

print(final_prompt)
