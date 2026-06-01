"""
协调器代理模式示例：Orchestrator 通过工具调用 ResearchAgent、WriteAgent、ReviewAgent。

与 11.py 的 AgentWorkflow 模式不同，本示例由顶层 Orchestrator 统一决策每一步，
子代理作为工具暴露，流程完全由协调器控制。
"""

import asyncio
import os
import re
from typing import Annotated

import httpx
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import Context, Workflow
from llama_index.llms.openai import OpenAI
from tavily import AsyncTavilyClient

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
tavily_api_key = os.environ["TAVILY_API_KEY"]
os.environ.setdefault("OPENAI_API_BASE", api_base)

Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)

_tavily_client = AsyncTavilyClient(
    api_key=tavily_api_key,
    client=httpx.AsyncClient(trust_env=False),
)


async def search_web(
    query: Annotated[str, "Search query for web research"],
) -> str:
    """Useful for using the web to answer questions."""
    return str(await _tavily_client.search(query))


research_agent = FunctionAgent(
    name="ResearchAgent",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic "
        "and record notes on the topic. You should output notes on the topic in a structured format."
    ),
    llm=Settings.llm,
    tools=[search_web],
    verbose=False,
)

write_agent = FunctionAgent(
    name="WriteAgent",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in markdown format. The content should be grounded in the research notes. "
        "Return your markdown report surrounded by <report>...</report> tags."
    ),
    llm=Settings.llm,
    tools=[],
    verbose=False,
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    system_prompt=(
        "You are the ReviewAgent that can review the written report and provide feedback. "
        "Your review should either approve the current report or request changes to be implemented."
    ),
    llm=Settings.llm,
    tools=[],
    verbose=False,
)


async def _run_sub_agent(agent: FunctionAgent, user_msg: str) -> str:
    sub_ctx = Context(agent)
    handler = Workflow.run(
        agent,
        ctx=sub_ctx,
        start_event=AgentWorkflowStartEvent(user_msg=user_msg),
    )
    return str(await handler)


async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await _run_sub_agent(
        research_agent,
        f"Write some notes about the following: {prompt}",
    )
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["research_notes"].append(str(result))
    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    async with ctx.store.edit_state() as ctx_state:
        notes = ctx_state["state"].get("research_notes", None)
        if not notes:
            return "No research notes to write from."

        user_msg = (
            "Write a markdown report from the following notes. "
            "Be sure to output the report in the following format: <report>...</report>:\n\n"
        )

        feedback = ctx_state["state"].get("review", None)
        if feedback:
            user_msg += f"<feedback>{feedback}</feedback>\n\n"

        notes_text = "\n\n".join(notes)
        user_msg += f"<research_notes>{notes_text}</research_notes>\n\n"

        result = await _run_sub_agent(write_agent, user_msg)
        match = re.search(r"<report>(.*)</report>", str(result), re.DOTALL)
        if not match:
            return "Write agent did not return a report in <report>...</report> format."
        report = match.group(1)
        ctx_state["state"]["report_content"] = str(report)

    return str(report)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    async with ctx.store.edit_state() as ctx_state:
        report = ctx_state["state"].get("report_content", None)
        if not report:
            return "No report content to review."

        result = await _run_sub_agent(
            review_agent,
            f"Review the following report: {report}",
        )
        ctx_state["state"]["review"] = str(result)

    return str(result)


orchestrator = FunctionAgent(
    name="Orchestrator",
    description="Useful for orchestrating the research, writing, and reviewing process.",
    tools=[call_research_agent, call_write_agent, call_review_agent],
    llm=Settings.llm,
    system_prompt=(
        "You are an expert in the field of report writing. "
        "You are given a user request and a list of tools that can help with the request. "
        "You are to orchestrate the tools to research, write, and review a report on the given topic. "
        "Once the review is positive, you should notify the user that the report is ready to be accessed."
    ),
    initial_state={
        "research_notes": [],
        "report_content": None,
        "review": None,
    },
    verbose=False,
)


async def run_orchestrator(user_msg: str, *, stream: bool = True) -> str:
    ctx = Context(orchestrator)
    handler = Workflow.run(
        orchestrator,
        ctx=ctx,
        start_event=AgentWorkflowStartEvent(user_msg=user_msg),
    )

    async for event in handler.stream_events():
        if not stream:
            continue
        if isinstance(event, AgentStream):
            if event.delta:
                print(event.delta, end="", flush=True)
        elif isinstance(event, AgentOutput) and event.tool_calls:
            print(
                "Planning to use tools:",
                [call.tool_name for call in event.tool_calls],
            )
        elif isinstance(event, ToolCall):
            print(f"Calling tool: {event.tool_name}")
        elif isinstance(event, ToolCallResult):
            print(f"Tool result ({event.tool_name}): {event.tool_output}")

    if stream:
        print()

    result = str(await handler)
    state = await ctx.store.get("state")
    print("\n--- Final report ---\n")
    print(state.get("report_content", result))
    return result


async def main() -> None:
    user_msg = "Write a report about the latest developments in the field of AI in 2026 with a focus on the latest trends and technologies. required write in Chinese."
    print(f"User: {user_msg}\n")
    await run_orchestrator(user_msg)


if __name__ == "__main__":
    asyncio.run(main())
