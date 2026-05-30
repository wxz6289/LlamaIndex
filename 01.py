from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]

# LlamaIndex defaults read OPENAI_API_BASE, not OPENAI_BASE_URL
os.environ.setdefault("OPENAI_API_BASE", api_base)

Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=api_key,
    api_base=api_base,
)

STORAGE_DIR = "storage"

def get_or_create_index() -> VectorStoreIndex:
    if os.path.isdir(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)

    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(STORAGE_DIR)
    return index

index = get_or_create_index()
# query_engine = index.as_query_engine(streaming=True)
query_engine = index.as_query_engine(similarity_top_k=5, streaming=True)
result = query_engine.query("What did the author do growing up?")
for chunk in result.response_gen:
    print(chunk, end="", flush=True)
