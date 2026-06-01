"""
Structured Output 示例：FunctionAgent 通过 output_cls 约束最终输出为 Pydantic 模型。

注意：await handler 返回的是 AgentOutput，其 __str__ 为自然语言回复；
结构化结果需通过 structured_response 或 get_pydantic_model() 获取。
"""

import asyncio
import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import Context, Workflow
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(".env")

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)
Settings.llm = llm


class MathResult(BaseModel):
    operation: str = Field(description="The performed operation, e.g. '3415 * 43144'")
    result: int = Field(description="The numeric result of the operation")


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


async def main() -> None:
    agent = FunctionAgent(
        tools=[multiply],
        name="calculator",
        system_prompt=(
            "You are a calculator agent who can multiply two numbers using the `multiply` tool. "
            "After using the tool, respond with the operation and its numeric result."
        ),
        output_cls=MathResult,
        llm=llm,
        verbose=True,
    )

    ctx = Context(agent)
    handler = Workflow.run(
        agent,
        ctx=ctx,
        start_event=AgentWorkflowStartEvent(
            user_msg="What is 3415 * 43144?",
        ),
    )
    response = await handler

    math_result = response.get_pydantic_model(MathResult)

    print("--- Natural language reply ---")
    print(response.response.content)
    print("\n--- Structured output (dict) ---")
    print(response.structured_response)
    print("\n--- Structured output (Pydantic) ---")
    print(math_result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
