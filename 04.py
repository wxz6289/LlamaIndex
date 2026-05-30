from llama_index.core import (
    Settings,
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import Context, Workflow
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]

# LlamaIndex defaults read OPENAI_API_BASE, not OPENAI_BASE_URL
os.environ.setdefault("OPENAI_API_BASE", api_base)

Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)

def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


workflow = FunctionAgent(
    tools=[multiply, add],
    llm=Settings.llm,
    system_prompt="You are a helpful assistant that can multiply and add two numbers.",
    verbose=True,
)

async def main():
    ctx = Context(workflow)
    handler = Workflow.run(
        workflow,
        ctx=ctx,
        start_event=AgentWorkflowStartEvent(
            user_msg="What is 21 ** 31? And what is 21 / 31?",
        ),
    )
    result = await handler
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
