from tavily import TavilyClient
import requests
import os

session = requests.Session()
session.trust_env = False

""" print(
    session.get(
        "https://api.tavily.com"
    )
) """

# print("KEY =", os.getenv("TAVILY_API_KEY"))

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY"),
    session=session,
)

result = client.search(
    query="Hangzhou weather today"
)

print(result)
