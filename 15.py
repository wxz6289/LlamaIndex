"""
Structured Output + 多 Agent 示例：MainAgent 调度 WeatherAgent 调用天气 API。

WeatherAgent 通过 Open-Meteo 获取实时天气，AgentWorkflow 将最终结果约束为 Weather 模型。
"""

import asyncio
import os
from typing import Annotated

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llama_index.core import Settings
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import Context, Workflow
from llama_index.llms.openai import OpenAI

load_dotenv(".env")

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)
Settings.llm = llm

# WMO weather code → human-readable description
WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class Weather(BaseModel):
    location: str = Field(description="The location")
    weather: str = Field(description="The weather summary")


async def get_weather(
    location: Annotated[str, "City name, e.g. 'Hangzhou' or 'Tokyo'"],
) -> str:
    """Get the current weather for a given location."""
    async with httpx.AsyncClient(trust_env=False, timeout=15.0) as client:
        geo_resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results") or []
        if not results:
            return f"Could not find location: {location}"

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        place_name = place.get("name", location)
        admin1 = place.get("admin1", "")
        country = place.get("country", "")
        full_location = ", ".join(part for part in (place_name, admin1, country) if part)

        weather_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            },
        )
        weather_resp.raise_for_status()
        current = weather_resp.json()["current"]

        condition = WMO_WEATHER_CODES.get(
            int(current["weather_code"]), "Unknown"
        )
        return (
            f"Location: {full_location}\n"
            f"Condition: {condition}\n"
            f"Temperature: {current['temperature_2m']}°C\n"
            f"Humidity: {current['relative_humidity_2m']}%\n"
            f"Wind speed: {current['wind_speed_10m']} km/h"
        )


weather_agent = FunctionAgent(
    llm=llm,
    tools=[get_weather],
    system_prompt=(
        "You are a weather agent that can get the weather for a given location. "
        "Always call the `get_weather` tool with the requested city before answering."
    ),
    name="WeatherAgent",
    description="The weather forecaster agent that queries live weather data.",
    can_handoff_to=["MainAgent"],
    verbose=False,
)

main_agent = FunctionAgent(
    name="MainAgent",
    tools=[],
    description="The main agent that dispatches weather queries to WeatherAgent.",
    system_prompt=(
        "You are the main agent. When the user asks about weather, "
        "hand off to WeatherAgent to fetch live data."
    ),
    can_handoff_to=["WeatherAgent"],
    llm=llm,
    verbose=False,
)

workflow = AgentWorkflow(
    agents=[main_agent, weather_agent],
    root_agent=main_agent.name,
    output_cls=Weather,
    verbose=False,
)


async def main() -> None:
    user_msg = "What is the weather in Hangzhou?"
    print(f"User: {user_msg}\n")

    ctx = Context(workflow)
    handler = Workflow.run(
        workflow,
        ctx=ctx,
        start_event=AgentWorkflowStartEvent(user_msg=user_msg),
    )
    response = await handler
    weather = response.get_pydantic_model(Weather)

    print("--- Natural language reply ---")
    print(response.response.content)
    print("\n--- Structured output (dict) ---")
    print(response.structured_response)
    print("\n--- Structured output (Pydantic) ---")
    print(weather.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
