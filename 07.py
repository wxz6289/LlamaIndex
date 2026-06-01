"""
Workflow Context 序列化示例：跨请求 / 跨进程恢复 Agent 多轮对话。

生产要点：
- 每轮对话结束后 to_dict 并持久化；下次请求 from_dict 恢复。
- serialize / deserialize 使用同一 Utf8JsonSerializer 实例。
- workflow 配置（tools、system_prompt、llm）在恢复前后必须一致。
- 落盘前压缩冗余字段，避免 user_msg_str / workers 等与 memory 重复。
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
from llama_index.core.workflow import Context, JsonSerializer, Workflow
from llama_index.llms.openai import OpenAI

load_dotenv(".env")

api_key = os.environ["OPENAI_API_KEY"]
api_base = os.environ["OPENAI_BASE_URL"]
model = os.environ["OPENAI_MODEL"]
os.environ.setdefault("OPENAI_API_BASE", api_base)

SESSION_DIR = Path("sessions")


class Utf8JsonSerializer(JsonSerializer):
    """JsonSerializer 子类：序列化时保留 UTF-8 中文，避免 \\uXXXX 转义。"""

    def serialize(self, value: object) -> str:
        try:
            serialized_value = self.serialize_value(value)
            return json.dumps(serialized_value, ensure_ascii=False)
        except Exception as exc:
            raise ValueError(
                f"Failed to serialize value: {type(value)}: {value!s}"
            ) from exc


SERIALIZER = Utf8JsonSerializer()


def build_workflow() -> FunctionAgent:
    return FunctionAgent(
        tools=[],
        llm=OpenAI(model=model, api_key=api_key, api_base=api_base),
        system_prompt="You are a helpful assistant that can answer questions.",
        verbose=True,
    )


def compact_context_for_storage(ctx_dict: dict[str, Any]) -> dict[str, Any]:
    """去掉与 memory 重复的运行时状态，仅保留恢复会话所需字段。"""
    compact = copy.deepcopy(ctx_dict)
    data = compact["state"]["state_data"]["_data"]
    memory = data.pop("memory")

    data.clear()
    data["memory"] = memory
    data["state"] = SERIALIZER.serialize({})
    data["max_iterations"] = SERIALIZER.serialize(20)
    data["early_stopping_method"] = SERIALIZER.serialize("force")
    data["formatted_input_with_state"] = SERIALIZER.serialize(False)

    compact["is_running"] = False
    compact["workers"] = {}
    return compact


def decode_context_for_file(ctx_dict: dict[str, Any]) -> dict[str, Any]:
    """将 state 内嵌的 JSON 字符串展开为对象，便于阅读且避免双重转义。"""
    decoded = copy.deepcopy(ctx_dict)
    data = decoded["state"]["state_data"]["_data"]
    decoded["state"]["state_data"]["_data"] = {
        key: json.loads(value) if isinstance(value, str) else value
        for key, value in data.items()
    }
    return decoded


def encode_context_from_file(
    stored: dict[str, Any],
    serializer: JsonSerializer = SERIALIZER,
) -> dict[str, Any]:
    """读取落盘 JSON 后，还原为 Context.from_dict 需要的序列化字符串格式。"""
    encoded = copy.deepcopy(stored)
    data = encoded["state"]["state_data"]["_data"]
    encoded["state"]["state_data"]["_data"] = {
        key: serializer.serialize(value) for key, value in data.items()
    }
    return encoded


def expand_context_for_restore(
    ctx_dict: dict[str, Any],
    serializer: JsonSerializer = SERIALIZER,
) -> dict[str, Any]:
    """补全 workflow 运行所需的默认 ephemeral 字段。"""
    expanded = copy.deepcopy(ctx_dict)
    data = expanded["state"]["state_data"]["_data"]
    defaults = {
        "num_iterations": serializer.serialize(0),
        "scratchpad": serializer.serialize([]),
        "current_tool_calls": serializer.serialize([]),
    }
    for key, value in defaults.items():
        data.setdefault(key, value)
    return expanded


def _is_decoded_file_format(stored: dict[str, Any]) -> bool:
    memory = stored.get("state", {}).get("state_data", {}).get("_data", {}).get(
        "memory"
    )
    return isinstance(memory, dict)


def prepare_context_payload(ctx_dict: dict[str, Any]) -> dict[str, Any]:
    return decode_context_for_file(compact_context_for_storage(ctx_dict))


def restore_context_payload(stored: dict[str, Any]) -> dict[str, Any]:
    if _is_decoded_file_format(stored):
        return expand_context_for_restore(encode_context_from_file(stored))
    return expand_context_for_restore(stored)


class SessionStore:
    """会话持久化：示例用本地 JSON 文件，生产可替换为 Redis / DB。"""

    def __init__(self, base_dir: Path = SESSION_DIR) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_")
        return self.base_dir / f"{safe_id}.json"

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, session_id: str, ctx_dict: dict[str, Any]) -> None:
        path = self._path(session_id)
        payload = json.dumps(
            prepare_context_payload(ctx_dict),
            ensure_ascii=False,
            indent=2,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.base_dir,
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            path.unlink()


class PersistentAgentSession:
    """封装 load → run → save，模拟 HTTP 每次请求独立加载 context。"""

    def __init__(
        self,
        workflow: FunctionAgent,
        store: SessionStore,
        session_id: str,
    ) -> None:
        self.workflow = workflow
        self.store = store
        self.session_id = session_id

    def _load_context(self) -> Context:
        stored = self.store.load(self.session_id)
        if stored is None:
            return Context(self.workflow)
        ctx_dict = restore_context_payload(stored)
        return Context.from_dict(
            self.workflow,
            ctx_dict,
            serializer=SERIALIZER,
        )

    def _save_context(self, ctx: Context) -> None:
        self.store.save(self.session_id, ctx.to_dict(serializer=SERIALIZER))

    async def chat(self, user_msg: str) -> str:
        ctx = self._load_context()
        handler = Workflow.run(
            self.workflow,
            ctx=ctx,
            start_event=AgentWorkflowStartEvent(user_msg=user_msg),
        )
        response = await handler
        self._save_context(ctx)
        return str(response)


async def main() -> None:
    store = SessionStore()
    session_id = "demo-user-001"

    store.delete(session_id)
    session = PersistentAgentSession(build_workflow(), store, session_id)

    print("=== 请求 1：自我介绍 ===")
    reply = await session.chat("你好，我的名字是张三，我今年20岁。")
    print(reply)

    print("\n=== 请求 2：跨请求恢复记忆 ===")
    session = PersistentAgentSession(build_workflow(), store, session_id)
    reply = await session.chat("我是谁？今年多少岁？")
    print(reply)

    print(f"\n会话已持久化: {store._path(session_id)}")


if __name__ == "__main__":
    asyncio.run(main())
