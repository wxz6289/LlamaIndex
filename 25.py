import asyncio
from llama_index.core.prompts import RichPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(".env")

llm = OpenAI(model=os.getenv("OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))

class SocialAccounts(BaseModel):
    instagram: Optional[str] = Field(default=None)
    bluesky: Optional[str] = Field(default=None)
    x: Optional[str] = Field(default=None)
    mastodon: Optional[str] = Field(default=None)

class User(BaseModel):
    name: str
    surname: str
    age: int
    email: str
    phone: str
    social_accounts: SocialAccounts



class ContactDetails(BaseModel):
    email: str
    phone: str
    social_accounts: SocialAccounts


async def main():
    template_str = "Please extract from the following XML code the contact details of the user:\n\n```xml\n{{ user | to_xml }}\n```\n\n"
    prompt = RichPromptTemplate(template_str)
    user = User(
        name="John",
        surname="Doe",
        age=30,
        email="john.doe@example.com",
        phone="123-456-7890",
        social_accounts={"bluesky": "john.doe", "instagram": "johndoe1234"},
    )
    sllm = llm.as_structured_llm(ContactDetails)
    structured_response = await sllm.achat(prompt.format_messages(user=user))
    print(structured_response.raw.email)
    print(structured_response.raw.phone)
    print(structured_response.raw.social_accounts.instagram)
    print(structured_response.raw.social_accounts.bluesky)

if __name__ == "__main__":
    asyncio.run(main())
