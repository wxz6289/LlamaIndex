"""
Multi-Agent 报告生成示例：ResearchAgent → WriteAgent → ReviewAgent 协作撰写报告。

流程：
1. ResearchAgent 使用 Tavily 搜索 Web，并通过 record_notes 记录笔记
2. WriteAgent 基于笔记调用 write_report 撰写 Markdown 报告
3. ReviewAgent 调用 review_report 审阅并反馈，必要时交还 WriteAgent 修订
"""

import asyncio
import os
from typing import Annotated

import httpx
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentStream,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
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


async def record_notes(
    ctx: Context,
    notes: Annotated[str, "Research notes to save"],
    notes_title: Annotated[str, "Title to save the notes under"],
) -> str:
    """Useful for recording notes on a given topic."""
    async with ctx.store.edit_state() as ctx_state:
        if "research_notes" not in ctx_state["state"]:
            ctx_state["state"]["research_notes"] = {}
        ctx_state["state"]["research_notes"][notes_title] = notes
    return "Notes recorded."


async def write_report(
    ctx: Context,
    report_content: Annotated[str, "Markdown formatted report content"],
) -> str:
    """Useful for writing a report on a given topic."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["report_content"] = report_content
    return "Report written."


async def review_report(
    ctx: Context,
    review: Annotated[str, "Review feedback for the report"],
) -> str:
    """Useful for reviewing a report and providing feedback."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["review"] = review
    return "Report reviewed."


def build_agents() -> tuple[FunctionAgent, FunctionAgent, FunctionAgent]:
    research_agent = FunctionAgent(
        name="ResearchAgent",
        description=(
            "Useful for searching the web for information on a given topic "
            "and recording notes on the topic."
        ),
        system_prompt=(
            "You are the ResearchAgent that can search the web for information on a given topic "
            "and record notes on the topic. "
            "Once notes are recorded and you are satisfied, you should hand off control to the "
            "WriteAgent to write a report on the topic. "
            "You should have at least some notes on a topic before handing off control to the WriteAgent."
        ),
        llm=Settings.llm,
        tools=[search_web, record_notes],
        can_handoff_to=["WriteAgent"],
        verbose=False,
    )

    write_agent = FunctionAgent(
        name="WriteAgent",
        description="Useful for writing a report on a given topic.",
        system_prompt=(
            "You are the WriteAgent that can write a report on a given topic. "
            "Your report should be in markdown format. The content should be grounded in the research notes. "
            "Once the report is written, you should get feedback at least once from the ReviewAgent."
        ),
        llm=Settings.llm,
        tools=[write_report],
        can_handoff_to=["ReviewAgent", "ResearchAgent"],
        verbose=False,
    )

    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Useful for reviewing a report and providing feedback.",
        system_prompt=(
            "You are the ReviewAgent that can review the written report and provide feedback. "
            "Your review should either approve the current report or request changes for the WriteAgent "
            "to implement. If you have feedback that requires changes, hand off control to the WriteAgent "
            "after submitting the review."
        ),
        llm=Settings.llm,
        tools=[review_report],
        can_handoff_to=["WriteAgent"],
        verbose=False,
    )

    return research_agent, write_agent, review_agent


def build_agent_workflow() -> AgentWorkflow:
    research_agent, write_agent, review_agent = build_agents()
    return AgentWorkflow(
        agents=[research_agent, write_agent, review_agent],
        root_agent=research_agent.name,
        initial_state={
            "research_notes": {},
            "report_content": "Not written yet.",
            "review": "Review required.",
        },
        verbose=False,
    )


async def run_report_workflow(user_msg: str, *, stream: bool = True) -> str:
    """运行多 Agent 报告工作流，可选流式打印工具调用与 Agent 切换。"""
    workflow = build_agent_workflow()
    ctx = Context(workflow)
    handler = workflow.run(
        ctx=ctx,
        user_msg=user_msg,
        max_iterations=20,
    )

    current_agent: str | None = None
    seen_tool_calls: set[str] = set()

    async for event in handler.stream_events():
        if stream:
            agent_name = getattr(event, "current_agent_name", None)
            if agent_name and agent_name != current_agent:
                current_agent = agent_name
                print(f"\n{'=' * 50}")
                print(f"Agent: {current_agent}")
                print(f"{'=' * 50}\n")

            if isinstance(event, AgentStream):
                if event.tool_calls:
                    for tc in event.tool_calls:
                        if tc.tool_id not in seen_tool_calls:
                            seen_tool_calls.add(tc.tool_id)
                            print(f"[calling tool: {tc.tool_name}]", flush=True)
                if event.delta:
                    print(event.delta, end="", flush=True)
            elif isinstance(event, AgentOutput) and event.response.content:
                print(f"\nOutput: {event.response.content}")
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
    user_msg = "请写一篇关于 Web3.0 的最新发展详细报告,内容涵盖技术、政策等，要求使用中文"
    print(f"User: {user_msg}\n")
    await run_report_workflow(user_msg)


if __name__ == "__main__":
    asyncio.run(main())
