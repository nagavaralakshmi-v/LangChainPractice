from langchain_core.runnables import RunnableLambda


def func(x):
    for y in x:
        yield str(y)


runnable = RunnableLambda(func)

for chunk in runnable.stream(range(5)):
    print(chunk)

# Async variant:
# async for chunk in await runnable.astream(range(5)):
#     print(chunk)