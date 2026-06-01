import asyncio
from typing import Annotated
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import Context, Workflow
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
os.environ.setdefault("OPENAI_API_BASE", api_base)

Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)

async def set_name(ctx: Context, name: Annotated[str, "The name of the user, must be a string, not an age or numeric value"]) -> str:
  if name.isdigit():
    raise ValueError(f"{name} looks like an age, not a name")
  async with ctx.store.edit_state() as ctx_state:
    ctx_state["state"]["name"] = name
  return f"Name set to {name}"

async def set_age(ctx: Context, age: Annotated[int, "The age of the user, must be an integer or a string that can be converted to an integer"]) -> str:
  try:
    age = int(age)
  except ValueError:
    raise ValueError(f"{age} looks like a name, not an age")
  async with ctx.store.edit_state() as ctx_state:
    ctx_state["state"]["age"] = age
  return f"Age set to {age}"

system_prompt = """You are a helpful assistant that can set the name of the user.
  Rules:
1. Only call set_name when the user explicitly provides a name.
2. Never use age as a name.
3. Ignore numbers when calling set_name.
4. If the user provides an age, answer normally without calling set_name.
  """

workflow = AgentWorkflow.from_tools_or_functions(
  [set_name, set_age],
  llm=Settings.llm,
  system_prompt=system_prompt,
  initial_state={"name": "unset", "age": "unset"},
  verbose=True,
)

async def main():
  ctx = Context(workflow)
  handler = Workflow.run(workflow, ctx=ctx, start_event=AgentWorkflowStartEvent(user_msg="我是谁？今年多少岁？"))
  response = await handler
  print(response)

  handler = Workflow.run(workflow, ctx=ctx, start_event=AgentWorkflowStartEvent(user_msg="你好，我的名字是King，今年26岁。"))
  response = await handler
  print(response)

  state = await ctx.store.get("state")
  print(state)
  name = state["name"]
  print(f"Name is {name}")
  age = state["age"]
  print(f"Age is {age}")

if __name__ == "__main__":
  asyncio.run(main())
