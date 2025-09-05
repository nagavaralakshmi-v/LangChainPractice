from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)  # Only last 2 interactions
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

print(conversation.run("I live in India."))
print(conversation.run("I love coding in Python."))
print(conversation.run("Where do I live?"))  # ❌ May forget if out of last 2 turns
