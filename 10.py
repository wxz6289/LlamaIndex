"""
Human-in-the-loop 示例：仅在执行危险任务时介入人工确认。

Agent 调用 dangerous_task 后，工具暂停等待 yes/no；
用户同意后，才真正执行指定脚本并返回输出。
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import (
    Context,
    HumanResponseEvent,
    InputRequiredEvent,
    Workflow,
)
from llama_index.llms.openai import OpenAI

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
os.environ.setdefault("OPENAI_API_BASE", api_base)

CONFIRM_USER = "King"
BASE_DIR = Path(__file__).resolve().parent

Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)


async def _run_python_file(file_path: Path) -> str:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(file_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(BASE_DIR),
        env=os.environ.copy(),
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace").strip()
    return (
        f"Executed {file_path.name} (exit code {proc.returncode}):\n"
        f"{output or '(no output)'}"
    )


async def dangerous_task(
    ctx: Context,
    file_path: Annotated[str, "Relative or absolute path to the Python script to execute"],
) -> str:
    """Execute a Python file after human confirmation."""
    target = Path(file_path)
    if not target.is_absolute():
        target = BASE_DIR / target
    if not target.is_file():
        return f"File not found: {file_path}"

    question = f"Confirm execution of {target.name}? (yes/no): "
    response = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_id=question,
        waiter_event=InputRequiredEvent(
            prefix=question,
            user_name=CONFIRM_USER,
        ),
        requirements={"user_name": CONFIRM_USER},
    )
    if response.response.strip().lower() != "yes":
        return "Dangerous task aborted."

    return await _run_python_file(target)


def build_workflow() -> FunctionAgent:
    return FunctionAgent(
        tools=[dangerous_task],
        llm=Settings.llm,
        system_prompt=(
            "You are a helpful assistant that performs dangerous tasks via the "
            "dangerous_task tool. When the user asks to execute a Python file, "
            "you MUST call dangerous_task with the file_path argument. "
            "Do not ask for confirmation yourself; the tool handles human approval."
        ),
        verbose=False,
    )


async def run_agent(user_msg: str) -> str:
    """运行 Agent；仅在收到 InputRequiredEvent 时采集人工确认。"""
    workflow = build_workflow()
    ctx = Context(workflow)
    handler = Workflow.run(
        workflow,
        ctx=ctx,
        start_event=AgentWorkflowStartEvent(
            user_msg=user_msg,
            max_iterations=10,
        ),
    )

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            response = input(event.prefix).strip()
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    user_name=event.user_name,
                )
            )

    return str(await handler)


async def main() -> None:
    user_msg = "I want to execute file test01.py and print the result"
    print(f"User: {user_msg}\n")
    result = await run_agent(user_msg)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
