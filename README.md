# LlamaIndex

LlamaIndex 学习示例：RAG、Agent、Yahoo Finance 工具等。

## 环境

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

```bash
uv sync
cp .env.example .env   # 填写 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL
```

## 脚本

| 文件 | 说明 |
|------|------|
| `01.py`–`05.py` | 分步示例 |
| `starter.py` | 文档 RAG + Agent |
| `test.py` | 向量索引查询 |

## 运行

```bash
uv run python 04.py
```

Yahoo 接口限流时可在 `.env` 设置兜底价格，例如 `STOCK_PRICE_NVDA=135.50`。
