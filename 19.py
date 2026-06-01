"""
Chroma 向量库示例：使用 ChromaDB 持久化本地向量索引并查询。

与 01.py（默认本地 storage）不同，本示例将向量存入 ChromaDB，
适合需要独立向量数据库或跨进程共享检索的场景。
"""

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv(".env")

DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "llamaindex-demo"
DEFAULT_QUERY = "What did the author do growing up?"


def configure_models() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ["OPENAI_BASE_URL"]
    model = os.environ["OPENAI_MODEL"]
    os.environ.setdefault("OPENAI_API_BASE", api_base)

    Settings.llm = OpenAI(model=model, api_key=api_key, api_base=api_base)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
        api_base=api_base,
    )


def get_or_create_index() -> VectorStoreIndex:
    """Load an existing Chroma collection or build it from local documents."""
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if collection.count() > 0:
        print(f"Loaded existing Chroma collection: {COLLECTION_NAME}")
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if not documents:
        raise ValueError(f"No documents found in {DATA_DIR}/")

    print(f"Building Chroma index from {DATA_DIR}/")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )


def main() -> None:
    configure_models()
    index = get_or_create_index()

    query = os.getenv("CHROMA_QUERY", DEFAULT_QUERY)
    print(f"Query: {query}\n")

    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(query)

    print("--- Answer ---")
    print(response)


if __name__ == "__main__":
    main()
