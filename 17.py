"""
Structured Output 示例（自定义解析）：通过 structured_output_fn 将对话历史转为 Pydantic 模型。

与 14.py 的 output_cls 不同，本示例在 agent 运行结束后，
用额外的 structured LLM 调用对完整对话做二次结构化解析，适合更复杂的转换逻辑。
"""

import asyncio
import os
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llama_index.core import Settings
from llama_index.core.agent.workflow import (
    AgentStreamStructuredOutput,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context, Workflow
from llama_index.llms.openai import OpenAI

load_dotenv(".env")

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)
Settings.llm = llm

ICE_CREAM_MENU: dict[str, dict[str, str]] = {
    "Gelato Italia": {
        "strawberry": "Strawberry with no extra sugar",
        "pistachio": "Sicilian pistachio with a hint of sea salt",
    },
    "Sweet Scoops": {
        "strawberry": "Strawberry cheesecake swirl with sugar",
        "vanilla": "Madagascar vanilla bean",
    },
}


class Flavor(BaseModel):
    flavor: str = Field(description="The ice cream flavor name")
    with_sugar: bool = Field(description="Whether the flavor contains added sugar")


async def structured_output_parsing(
    messages: list[ChatMessage],
) -> dict[str, Any]:
    """Parse agent conversation history into a Flavor dict via structured LLM."""
    sllm = llm.as_structured_llm(Flavor)
    parse_messages = list(messages)
    parse_messages.append(
        ChatMessage(
            role="user",
            content=(
                "Given the previous message history, extract the ice cream flavor "
                "and whether it contains added sugar. Respond using the required format."
            ),
        )
    )
    response = await sllm.achat(parse_messages)
    return response.raw.model_dump()


def get_flavor(
    ice_cream_shop: Annotated[str, "Name of the ice cream shop"],
    flavor_name: Annotated[str, "Flavor to look up, e.g. 'strawberry'"],
) -> str:
    """Look up a flavor description for a given shop and flavor name."""
    shop_menu = ICE_CREAM_MENU.get(ice_cream_shop)
    if not shop_menu:
        return f"No menu found for shop: {ice_cream_shop}"

    flavor_key = flavor_name.strip().lower()
    description = shop_menu.get(flavor_key)
    if not description:
        available = ", ".join(shop_menu)
        return (
            f"Flavor '{flavor_name}' not found at {ice_cream_shop}. "
            f"Available flavors: {available}"
        )
    return description


agent = FunctionAgent(
    tools=[get_flavor],
    name="ice_cream_shopper",
    system_prompt=(
        "You are an agent that knows the ice cream flavors in various shops. "
        "Use the `get_flavor` tool to look up flavors before answering."
    ),
    structured_output_fn=structured_output_parsing,
    llm=llm,
    verbose=False,
)


async def main() -> None:
    user_msg = "What strawberry flavor is available at Gelato Italia?"
    print(f"User: {user_msg}\n")

    ctx = Context(agent)
    handler = Workflow.run(
        agent,
        ctx=ctx,
        start_event=AgentWorkflowStartEvent(user_msg=user_msg),
    )

    async for event in handler.stream_events():
        if isinstance(event, AgentStreamStructuredOutput):
            print("--- Streaming structured output ---")
            print(event.output)
        elif isinstance(event, ToolCall):
            print(f"Calling tool: {event.tool_name} {event.tool_kwargs}")
        elif isinstance(event, ToolCallResult):
            print(f"Tool result: {event.tool_output}")

    response = await handler
    flavor = response.get_pydantic_model(Flavor)

    print("\n--- Natural language reply ---")
    print(response.response.content)
    print("\n--- Structured output (dict) ---")
    print(response.structured_response)
    print("\n--- Structured output (Pydantic) ---")
    print(flavor.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
