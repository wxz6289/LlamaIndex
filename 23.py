"""
Direct Prompting 示例：手写提示词 + JSON Schema，自行解析 LLM 输出。

与 20/21/22 不同，本示例不依赖 structured LLM、structured_predict 或 tool calling，
直接将 schema 和抽取规则写入 prompt，再用 Pydantic 校验 JSON。
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from pydantic import BaseModel, Field, field_validator

load_dotenv(".env")

INVOICE_PDF = Path("resources/invoice.pdf")

GENERIC_UNIT_LABELS = frozenset(
    {"项", "件", "个", "次", "月", "年", "服务费", "服务", "台", "套", "批"}
)


class LineItem(BaseModel):
    """A line item in an invoice."""

    item_name: str = Field(
        description="项目名称。仅来自项目名称列，合并换行后的完整名称。"
    )
    price: float = Field(description="该行金额（金额列），不含税。")

    @field_validator("item_name")
    @classmethod
    def strip_item_name(cls, value: str) -> str:
        return value.strip()


class Invoice(BaseModel):
    """A representation of information from an invoice."""

    invoice_id: str = Field(description="发票号码。找不到时可留空字符串。")
    date: datetime = Field(description="开票日期")
    line_items: list[LineItem] = Field(
        description="发票明细行，每一行表格记录对应一个元素。"
    )


def configure_llm() -> OpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ["OPENAI_BASE_URL"]
    model = os.environ["OPENAI_MODEL"]
    os.environ.setdefault("OPENAI_API_BASE", api_base)

    return OpenAI(model=model, api_key=api_key, api_base=api_base)


def load_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = PDFReader().load_data(file=pdf_path)
    if not documents:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    return documents[0].text


def build_prompt(pdf_text: str) -> str:
    schema = json.dumps(Invoice.model_json_schema(), indent=2, ensure_ascii=False)
    return (
        "Here is a JSON schema for an invoice:\n"
        f"{schema}\n\n"
        "Extract invoice information from the following Chinese e-invoice text.\n"
        "Rules:\n"
        "- Each table row is ONE line item.\n"
        "- item_name must come ONLY from the 项目名称 column. Merge broken lines.\n"
        "- Values in the 单位 column (e.g. 服务费, 项) are NOT separate line items.\n"
        "- price is the 金额 column value (pre-tax).\n"
        "- invoice_id is the 发票号码 when present.\n"
        "Format your output as a JSON object according to the schema above.\n"
        "Do not include any other text than the JSON object.\n"
        "Omit any markdown formatting. Do not include any preamble or explanation.\n\n"
        f"{pdf_text}"
    )


def parse_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def normalize_line_items(line_items: list[LineItem]) -> list[LineItem]:
    if len(line_items) <= 1:
        return line_items

    filtered: list[LineItem] = []
    for item in line_items:
        name = item.item_name.strip().lstrip("*").strip()
        if name in GENERIC_UNIT_LABELS:
            continue
        if len(name) <= 2 and not re.search(r"[\u4e00-\u9fff]{2,}", name):
            continue
        filtered.append(item)

    if not filtered:
        return line_items

    by_amount: dict[float, LineItem] = {}
    for item in filtered:
        current = by_amount.get(item.price)
        if current is None or len(item.item_name) > len(current.item_name):
            by_amount[item.price] = item

    return list(by_amount.values())


def fill_invoice_id(invoice: Invoice, pdf_text: str) -> Invoice:
    match = re.search(r"\b(\d{20})\b", pdf_text)
    if match:
        invoice.invoice_id = match.group(1)
    return invoice


def extract_invoice_with_prompt(llm: OpenAI, pdf_text: str) -> Invoice:
    prompt = build_prompt(pdf_text)
    response = llm.complete(prompt)
    json_text = parse_json_response(response.text)
    invoice = Invoice.model_validate_json(json_text)
    invoice.line_items = normalize_line_items(invoice.line_items)
    return fill_invoice_id(invoice, pdf_text)


def main() -> None:
    llm = configure_llm()
    pdf_path = Path(os.getenv("INVOICE_PDF_PATH", INVOICE_PDF))

    print(f"Reading PDF: {pdf_path}")
    pdf_text = load_pdf_text(pdf_path)

    print("Extracting via direct prompting...\n")
    invoice = extract_invoice_with_prompt(llm, pdf_text)

    print("--- Pydantic object ---")
    print(invoice)
    print("\n--- JSON ---")
    print(json.dumps(invoice.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
