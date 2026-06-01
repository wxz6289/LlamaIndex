"""
自定义多智能体模式示例：PlannerWorkflow 用 LLM 生成 XML 计划，再按步骤调用子代理。

与 11.py（AgentWorkflow 自动 handoff）、12.py（Orchestrator 工具调度）不同，
本示例由自定义 Workflow 完全掌控规划与执行循环，可插入任意业务逻辑。
"""

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from typing import Annotated, Any, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI

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

INITIAL_STATE: dict[str, Any] = {
    "research_notes": [],
    "report_content": "Not written yet.",
    "review": "Review required.",
}

PLANNER_PROMPT = """You are a planner chatbot.

Given a user request and the current state, break the solution into ordered <step> blocks.
Each step must specify the agent to call and the message to send, e.g.
<plan>
  <step agent="ResearchAgent">search for …</step>
  <step agent="WriteAgent">draft a report …</step>
  <step agent="ReviewAgent">review the report …</step>
</plan>

<state>
{state}
</state>

<available_agents>
{available_agents}
</available_agents>

The general flow should be:
- Record research notes
- Write a report
- Review the report
- Write the report again if the review is not positive enough

If the user request does not require any steps, you can skip the <plan> block and respond directly.
"""


async def search_web(
    query: Annotated[str, "Search query for web research"],
) -> str:
    """Useful for using the web to answer questions."""
    return str(await _tavily_client.search(query))


research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for recording research notes based on a specific prompt.",
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
    description="Useful for writing a report based on research notes or revising based on feedback.",
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
    description="Useful for reviewing a report and providing feedback.",
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
    """Record research notes and append them to shared workflow state."""
    result = await _run_sub_agent(
        research_agent,
        f"Write some notes about the following: {prompt}",
    )
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["research_notes"].append(str(result))
    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Write or revise a report from research notes and review feedback."""
    async with ctx.store.edit_state() as ctx_state:
        notes = ctx_state["state"].get("research_notes")
        if not notes:
            return "No research notes to write from."

        user_msg = (
            "Write a markdown report from the following notes. "
            "Be sure to output the report in the following format: <report>...</report>:\n\n"
        )
        feedback = ctx_state["state"].get("review")
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
    """Review the current report and store feedback in shared workflow state."""
    async with ctx.store.edit_state() as ctx_state:
        report = ctx_state["state"].get("report_content")
        if not report or report == "Not written yet.":
            return "No report content to review."

        result = await _run_sub_agent(
            review_agent,
            f"Review the following report: {report}",
        )
        ctx_state["state"]["review"] = str(result)

    return str(result)


class InputEvent(StartEvent):
    user_msg: Optional[str] = Field(default=None)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    state: Optional[dict[str, Any]] = Field(default=None)


class OutputEvent(StopEvent):
    response: str
    chat_history: list[ChatMessage]
    state: dict[str, Any]


class StreamEvent(Event):
    delta: str


class PlanEvent(Event):
    step_info: str


class PlanStep(BaseModel):
    agent_name: str
    agent_input: str


class Plan(BaseModel):
    steps: list[PlanStep]


class ExecuteEvent(Event):
    plan: Plan
    chat_history: list[ChatMessage]


class PlannerWorkflow(Workflow):
    llm: OpenAI = Settings.llm
    agents: dict[str, FunctionAgent] = {
        "ResearchAgent": research_agent,
        "WriteAgent": write_agent,
        "ReviewAgent": review_agent,
    }

    @step
    async def plan(
        self, ctx: Context, ev: InputEvent
    ) -> ExecuteEvent | OutputEvent:
        if ev.state:
            await ctx.store.set("state", ev.state)
        elif not await ctx.store.get("state", default=None):
            await ctx.store.set("state", INITIAL_STATE.copy())

        chat_history = list(ev.chat_history)
        if ev.user_msg:
            chat_history.append(ChatMessage(role="user", content=ev.user_msg))

        state = await ctx.store.get("state")
        available_agents_str = "\n".join(
            f'<agent name="{agent.name}">{agent.description}</agent>'
            for agent in self.agents.values()
        )
        system_prompt = ChatMessage(
            role="system",
            content=PLANNER_PROMPT.format(
                state=str(state),
                available_agents=available_agents_str,
            ),
        )

        response = await self.llm.astream_chat(messages=[system_prompt] + chat_history)
        full_response = ""
        async for chunk in response:
            full_response += chunk.delta or ""
            if chunk.delta:
                ctx.write_event_to_stream(StreamEvent(delta=chunk.delta))

        xml_match = re.search(r"(<plan>.*</plan>)", full_response, re.DOTALL)
        if not xml_match:
            chat_history.append(
                ChatMessage(role="assistant", content=full_response)
            )
            return OutputEvent(
                response=full_response,
                chat_history=chat_history,
                state=state,
            )

        root = ET.fromstring(xml_match.group(1))
        plan = Plan(steps=[])
        for plan_step in root.findall("step"):
            plan.steps.append(
                PlanStep(
                    agent_name=plan_step.attrib["agent"],
                    agent_input=plan_step.text.strip() if plan_step.text else "",
                )
            )
        return ExecuteEvent(plan=plan, chat_history=chat_history)

    @step
    async def execute(self, ctx: Context, ev: ExecuteEvent) -> InputEvent:
        chat_history = list(ev.chat_history)

        for plan_step in ev.plan.steps:
            ctx.write_event_to_stream(
                PlanEvent(
                    step_info=(
                        f'<step agent="{plan_step.agent_name}">'
                        f"{plan_step.agent_input}</step>"
                    ),
                ),
            )

            if plan_step.agent_name == "ResearchAgent":
                await call_research_agent(ctx, plan_step.agent_input)
            elif plan_step.agent_name == "WriteAgent":
                await call_write_agent(ctx)
            elif plan_step.agent_name == "ReviewAgent":
                await call_review_agent(ctx)
            else:
                raise ValueError(f"Unknown agent: {plan_step.agent_name}")

        state = await ctx.store.get("state")
        chat_history.append(
            ChatMessage(
                role="user",
                content=(
                    "I've completed the previous steps, here's the updated state:\n\n"
                    f"<state>\n{state}\n</state>\n\n"
                    "Do you need to continue and plan more steps? "
                    "If not, write a final response."
                ),
            )
        )
        return InputEvent(chat_history=chat_history)


async def run_planner_workflow(user_msg: str, *, stream: bool = True) -> OutputEvent:
    """Run the custom planner workflow with optional streaming."""
    workflow = PlannerWorkflow(timeout=None)
    handler = workflow.run(
        user_msg=user_msg,
        chat_history=[],
        state=INITIAL_STATE.copy(),
    )

    async for event in handler.stream_events():
        if not stream:
            continue
        if isinstance(event, StreamEvent):
            if event.delta:
                print(event.delta, end="", flush=True)
        elif isinstance(event, PlanEvent):
            print(f"\nExecuting plan step: {event.step_info}")

    if stream:
        print()

    result: OutputEvent = await handler
    state = await handler.ctx.store.get("state")
    print("\n--- Planner response ---\n")
    print(result.response)
    print("\n--- Final report ---\n")
    print(state.get("report_content", "(no report)"))
    return result


async def main() -> None:
    user_msg = (
        "Write a report about the latest developments in the field of frontend development in 2026, "
        "focusing on the latest trends and technologies. Write in Chinese."
    )
    print(f"User: {user_msg}\n")
    await run_planner_workflow(user_msg)


if __name__ == "__main__":
    asyncio.run(main())
