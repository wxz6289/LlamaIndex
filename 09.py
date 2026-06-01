from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import AgentStream, FunctionAgent
from llama_index.core.workflow import Context, Workflow
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
import asyncio
import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import requests
from tavily import TavilyClient

session = requests.Session()
session.trust_env = False

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
tavily_api_key = os.environ["TAVILY_API_KEY"]
os.environ.setdefault("OPENAI_API_BASE", api_base)

Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)

tavily_spec = TavilyToolSpec(api_key=tavily_api_key)
tavily_spec.client = TavilyClient(api_key=tavily_api_key, session=session)
tools = tavily_spec.to_tool_list()

workflow = FunctionAgent(
  tools=tools,
  llm=Settings.llm,
  system_prompt="You are a helpful assistant that can search the web for information.",
  verbose=True,
)

ctx = Context(workflow)

async def main():
  handler = Workflow.run(
    workflow,
    ctx=ctx,
    start_event=AgentWorkflowStartEvent(
      user_msg="What is the weather in Hangzhou today?",
      max_iterations=5,
    ),
  )
  seen_tool_calls: set[str] = set()
  async for event in handler.stream_events():
    if isinstance(event, AgentStream):
      if event.tool_calls:
        for tc in event.tool_calls:
          if tc.tool_id not in seen_tool_calls:
            seen_tool_calls.add(tc.tool_id)
            print(f"\n[calling tool: {tc.tool_name}]", flush=True)
      print(event.delta, end="", flush=True)
  print()
  result = await handler
  print(result)

if __name__ == "__main__":
  asyncio.run(main())
