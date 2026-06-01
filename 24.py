import os
from pydantic import BaseModel
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI
from typing import Dict
from dotenv import load_dotenv

load_dotenv(".env")

template_str = "Please extract from the following XML code the contact details of the user:\n\n```xml\n{{ user | to_xml }}\n```\n\n"
prompt = RichPromptTemplate(template_str)


class User(BaseModel):
    name: str
    surname: str
    age: int
    email: str
    phone: str
    social_accounts: Dict[str, str]


user = User(
    name="John",
    surname="Doe",
    age=30,
    email="john.doe@example.com",
    phone="123-456-7890",
    social_accounts={"bluesky": "john.doe", "instagram": "johndoe1234"},
)

## check how the prompt would look like

prompt.format(user=user)

llm = OpenAI(model=os.getenv("OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))

response = llm.chat(prompt.format_messages(user=user))

print(response.message.content)
