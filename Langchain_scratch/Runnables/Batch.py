from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: str(x))
runnable.batch([7, 8, 9])

# Async variant:
# await runnable.abatch([7, 8, 9])