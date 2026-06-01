from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import Context, Workflow
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
os.environ.setdefault("OPENAI_API_BASE", api_base)

workflow = FunctionAgent(
    tools=[],
    llm=OpenAI(model=model, api_key=api_key, api_base=api_base),
    system_prompt="You are a helpful assistant that can answer questions.",
    verbose=True,
)

ctx = Context(workflow)

async def main():
    handler = Workflow.run(workflow, ctx=ctx, start_event=AgentWorkflowStartEvent(user_msg="你好，我的名字是张三，我今年20岁。"))
    response = await handler
    print(response)
    handler = Workflow.run(workflow, ctx=ctx, start_event=AgentWorkflowStartEvent(user_msg="我是谁？今年多少岁？"))
    response = await handler
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
