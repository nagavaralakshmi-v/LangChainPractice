from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: str(x))
print(runnable.invoke(5))

# Async variant:
# await runnable.ainvoke(5)