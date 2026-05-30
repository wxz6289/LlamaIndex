from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
import asyncio
import os

load_dotenv('.env')

def multiply(a: float, b: float) -> float:
  """Multiply two numbers"""
  return a * b

agent = FunctionAgent(
  tools=[multiply],
  llm=OpenAI(model=os.environ["OPENAI_MODEL"], api_key=os.environ["OPENAI_API_KEY"], api_base=os.environ["OPENAI_BASE_URL"]),
  system_prompt="You are a helpful assistant that can multiply two numbers.",
  verbose=True,
)

async def main():
  ctx = Context(agent)
  result = await agent.run("My name is King and I am 25 years old.", ctx=ctx)  # pyright: ignore[reportDeprecated]
  print(result)
  result = await agent.run("What is my name?", ctx=ctx)
  print(result)
  result = await agent.run("What is my age?", ctx=ctx)
  print(result)

if __name__ == "__main__":
    asyncio.run(main())
