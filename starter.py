from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
os.environ.setdefault("OPENAI_API_BASE", api_base)

STORAGE_DIR = "storage"

llm = OpenAI(model=model, api_key=api_key, api_base=api_base)
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=api_key,
    api_base=api_base,
)


def get_or_create_index() -> VectorStoreIndex:
    if os.path.isdir(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(STORAGE_DIR)
    return index


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def main() -> None:
    index = get_or_create_index()
    query_engine = index.as_query_engine()

    async def search_documents(query: str) -> str:
        """Search the documents for the query."""
        response = await query_engine.aquery(query)
        return str(response)

    agent = FunctionAgent(
        tools=[multiply, search_documents],
        llm=llm,
        system_prompt=(
            "You are a helpful assistant that can search documents and do math. "
            "Use search_documents for questions about the essay."
        ),
        verbose=True,
    )

    ctx = Context(agent)
    result = await agent.run(
        "What did the author do in college? Also, what's 32**4?",
        ctx=ctx,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
