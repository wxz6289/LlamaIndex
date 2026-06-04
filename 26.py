"""
Tracing & Debugging 示例：LlamaIndex 可观测性入门。

演示四种常用的跟踪与调试方式：
1. debug  — LlamaDebugHandler：打印事件调用链与各阶段耗时
2. tokens — TokenCountingHandler：统计 LLM / Embedding token 用量
3. simple — set_global_handler("simple")：打印每次 LLM 调用的输入与输出
4. inst   — instrumentation 模块：监听 LLM 事件（新 API，逐步替代 callbacks）

运行前请先执行 01.py 构建 storage/ 索引，或确保 data/ 目录存在。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import tiktoken
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    set_global_handler,
)
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, TokenCountingHandler
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMChatStartEvent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv(".env")

STORAGE_DIR = "storage"
DATA_DIR = "data"
DEFAULT_QUERY = "What did the author do growing up?"
MODES = ("debug", "tokens", "simple", "inst")


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
    if os.path.isdir(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if not documents:
        raise ValueError(f"No documents found in {DATA_DIR}/. Run 01.py first.")

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(STORAGE_DIR)
    return index


def enable_debug_logging() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    logging.getLogger("llama_index").setLevel(logging.DEBUG)


def setup_debug_handler() -> LlamaDebugHandler:
    """LlamaDebugHandler：自动打印 trace 树与各阶段耗时。"""
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([debug_handler])
    return debug_handler


def setup_token_counter() -> TokenCountingHandler:
    """TokenCountingHandler：统计 prompt / completion / embedding token。"""
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
        verbose=False,
    )
    Settings.callback_manager = CallbackManager([token_counter])
    return token_counter


def setup_simple_handler() -> None:
    """一行启用：每次 LLM 调用时打印 Messages 与 Response。"""
    set_global_handler("simple")
    # global_handler 会在创建 CallbackManager 时自动挂载
    Settings.callback_manager = CallbackManager()


class LLMEventPrinter(BaseEventHandler):
    """instrumentation 示例：监听 LLM 调用开始与结束事件。"""

    @classmethod
    def class_name(cls) -> str:
        return "LLMEventPrinter"

    def handle(self, event: Any, **kwargs: Any) -> None:
        if isinstance(event, LLMChatStartEvent):
            model = event.model_dict.get("model", "unknown")
            print(f"\n[instrumentation] LLM start  model={model}")
        elif isinstance(event, LLMChatEndEvent):
            usage = getattr(event.response.raw, "usage", None)
            if usage is not None:
                print(
                    "[instrumentation] LLM end    "
                    f"prompt={usage.prompt_tokens} "
                    f"completion={usage.completion_tokens}"
                )
            else:
                print("[instrumentation] LLM end")


def setup_instrumentation() -> None:
    """instrumentation 模块：基于 EventHandler 的新式可观测性 API。"""
    dispatcher = get_dispatcher()
    dispatcher.add_event_handler(LLMEventPrinter())


def print_token_summary(token_counter: TokenCountingHandler) -> None:
    print("\n--- Token usage ---")
    print(f"LLM prompt tokens:     {token_counter.prompt_llm_token_count}")
    print(f"LLM completion tokens: {token_counter.completion_llm_token_count}")
    print(f"LLM total tokens:      {token_counter.total_llm_token_count}")
    print(f"Embedding tokens:      {token_counter.total_embedding_token_count}")


def print_debug_summary(debug_handler: LlamaDebugHandler) -> None:
    print("\n--- Event summary ---")
    for event_type, pairs in debug_handler.event_pairs_by_type.items():
        print(f"{event_type.name:12s}  events={len(pairs)}")

    stats = debug_handler.get_event_time_info()
    print(
        f"\nTotal tracked time: {stats.total_secs:.3f}s "
        f"({stats.total_count} events, avg {stats.average_secs:.3f}s)"
    )


def run_query(mode: str, query: str, verbose_logging: bool) -> None:
    if verbose_logging:
        enable_debug_logging()
        print("=== Debug logging enabled (llama_index.*) ===\n")

    debug_handler: LlamaDebugHandler | None = None
    token_counter: TokenCountingHandler | None = None

    if mode == "debug":
        print("=== Mode: LlamaDebugHandler ===\n")
        debug_handler = setup_debug_handler()
    elif mode == "tokens":
        print("=== Mode: TokenCountingHandler ===\n")
        token_counter = setup_token_counter()
    elif mode == "simple":
        print("=== Mode: set_global_handler('simple') ===\n")
        setup_simple_handler()
    elif mode == "inst":
        print("=== Mode: instrumentation (LLM events) ===\n")
        setup_instrumentation()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    configure_models()
    index = get_or_create_index()

    print(f"Query: {query}\n")
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)

    print("\n--- Answer ---")
    print(response)

    if token_counter is not None:
        print_token_summary(token_counter)
    if debug_handler is not None:
        print_debug_summary(debug_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex tracing & debugging demo")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="debug",
        help="debug=事件链路, tokens=token统计, simple=LLM输入输出, inst=instrumentation",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query text")
    parser.add_argument(
        "--verbose-logging",
        action="store_true",
        help="Enable DEBUG logging for llama_index (works with any mode)",
    )
    args = parser.parse_args()

    run_query(args.mode, args.query, args.verbose_logging)


if __name__ == "__main__":
    main()
