# LlamaIndex

LlamaIndex 学习示例子项目：RAG、Agent、多智能体、结构化抽取、LlamaCloud、Chroma 与可观测性等。

依赖与运行统一使用 **[uv](https://docs.astral.sh/uv/)**。

## 环境

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

```bash
cd LlamaIndex
uv sync
cp .env.example .env   # OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL 等
```

可选：用 Cursor/VS Code 打开 [`LlamaIndex.code-workspace`](./LlamaIndex.code-workspace)。

## 项目结构

```text
LlamaIndex/
├── 01.py–26.py          # 编号学习脚本（建议按序阅读）
├── starter.py           # 文档 RAG + Agent 入门
├── main.py              # 早期综合示例
├── data/                # 示例文本（如 paul_graham_essay.txt）
├── resources/           # 示例 PDF（invoice.pdf）；大体积参考书请放本地勿提交
├── doc/                 # 学习笔记
├── storage/             # 本地向量索引持久化（git 忽略）
├── chroma_db/           # Chroma 持久化（git 忽略）
├── pyproject.toml
└── uv.lock
```

## 脚本索引

| 文件 | 主题 |
|------|------|
| `01.py` | 文档加载、向量索引、持久化 `storage/` |
| `02.py`–`04.py` | RAG 查询与配置进阶 |
| `05.py` | Yahoo Finance 工具 + FunctionAgent |
| `06.py` | Agent Workflow 基础 |
| `07.py` | Workflow / Agent 综合练习 |
| `08.py` | 多 Agent（AgentWorkflow handoff） |
| `09.py` | Tavily 搜索 + Agent |
| `10.py` | Human-in-the-loop（危险操作需确认） |
| `11.py` | Multi-Agent 报告（Research → Write → Review） |
| `12.py` | Orchestrator 协调子 Agent |
| `13.py` | 自定义 Planner Workflow（XML 计划） |
| `14.py` | Structured Output（`output_cls`） |
| `15.py` | 多 Agent + 天气 API 结构化输出 |
| `16.py`–`17.py` | `structured_output_fn` 自定义解析 |
| `18.py` | LlamaCloud 托管索引检索 |
| `19.py` | Chroma 向量库持久化 |
| `20.py`–`23.py` | 发票 PDF 结构化抽取（多种 API 对比） |
| `24.py`–`25.py` | RichPromptTemplate 与异步提示 |
| `26.py` | Tracing & Debugging（debug / tokens / simple / inst） |
| `starter.py` | RAG + Agent 快速上手 |
| `test.py` | 向量索引查询 smoke test |

## 运行

```bash
# 首次建议从 01 构建索引
uv run python 01.py

# 任意示例
uv run python 26.py --mode debug
uv run python 26.py --mode tokens --query "What did the author do growing up?"
```

`26.py` 依赖 `storage/` 或 `data/` 中已有文档；若无索引请先运行 `01.py`。

Yahoo 接口限流时可在 `.env` 设置兜底价格，例如 `STOCK_PRICE_NVDA=135.50`。

## 相关资源

- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- 父 monorepo：[python](../README.md)（本目录为 git submodule）
