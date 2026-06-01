"""
LlamaCloud 索引查询示例：连接托管索引，执行检索与 RAG 问答。

流程：
1. 连接已有 LlamaCloud 索引（不存在时可从 data/ 自动创建）
2. 使用 as_retriever() 检索相关节点
3. 使用 as_query_engine() 生成带上下文的回答
"""

import os

from dotenv import load_dotenv
from llama_cloud_services import LlamaCloudIndex
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv(".env")

DATA_DIR = "data"
INDEX_NAME = os.getenv("LLAMA_CLOUD_INDEX_NAME", "llamaindex-demo")
PROJECT_NAME = os.getenv("LLAMA_CLOUD_PROJECT_NAME", "Default")
DEFAULT_QUERY = "What did the author do growing up?"


def get_cloud_api_key() -> str:
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_INDEX_CLOUD_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing LlamaCloud API key. Set LLAMA_CLOUD_API_KEY or "
            "LLAMA_INDEX_CLOUD_API_KEY in .env"
        )
    return api_key


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


def get_or_create_index(*, create_if_missing: bool = True) -> LlamaCloudIndex:
    """Connect to an existing LlamaCloud index, optionally creating it from local data."""
    api_key = get_cloud_api_key()

    try:
        index = LlamaCloudIndex(
            name=INDEX_NAME,
            project_name=PROJECT_NAME,
            api_key=api_key,
        )
        print(f"Connected to existing LlamaCloud index: {INDEX_NAME}")
        return index
    except ValueError as exc:
        if not create_if_missing or "Unknown index name" not in str(exc):
            raise

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if not documents:
        raise ValueError(f"No documents found in {DATA_DIR}/")

    print(f"Creating LlamaCloud index: {INDEX_NAME}")
    index = LlamaCloudIndex.from_documents(
        documents=documents,
        name=INDEX_NAME,
        project_name=PROJECT_NAME,
        api_key=api_key,
        verbose=True,
    )
    index.wait_for_completion()
    print("Index ingestion completed.")
    return index


def run_retrieval(index: LlamaCloudIndex, query: str) -> None:
    retriever = index.as_retriever(
        dense_similarity_top_k=3,
        enable_reranking=True,
        rerank_top_n=3,
    )
    nodes = retriever.retrieve(query)

    print(f"\n--- Retrieval results ({len(nodes)} nodes) ---")
    for i, node in enumerate(nodes, start=1):
        preview = node.node.get_content().replace("\n", " ")[:160]
        print(f"{i}. score={node.score:.4f} | {preview}...")


def run_query(index: LlamaCloudIndex, query: str) -> None:
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        streaming=False,
    )
    response = query_engine.query(query)

    print("\n--- Query engine answer ---")
    print(response)


def main() -> None:
    configure_models()
    index = get_or_create_index()

    query = os.getenv("LLAMA_CLOUD_QUERY", DEFAULT_QUERY)
    print(f"Query: {query}")

    run_retrieval(index, query)
    run_query(index, query)


if __name__ == "__main__":
    main()
