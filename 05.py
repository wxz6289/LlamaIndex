from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context, Workflow
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
import asyncio
import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
os.environ.setdefault("OPENAI_API_BASE", api_base)


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = YahooFinanceToolSpec().to_tool_list()
tools.append(multiply)

Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)

workflow = FunctionAgent(
    tools=tools,
    llm=Settings.llm,
    system_prompt=(
        "You are a helpful assistant that can search for stock information "
        "and multiply two numbers."
    ),
    verbose=True,
)


async def main():
    ctx = Context(workflow)
    handler = Workflow.run(
        workflow,
        ctx=ctx,
        start_event=AgentWorkflowStartEvent(
            user_msg="What is the stock price of Nvidia multiplied by 12?",
            max_iterations=5,
            early_stopping_method="generate",
        ),
    )
    result = await handler
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
