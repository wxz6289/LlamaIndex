"""
Tool Calling 示例：通过 llm.chat_with_tools 直接调用 Pydantic 工具提取结构化数据。

与 20.py（as_structured_llm）、21.py（structured_predict）不同，
本示例将 Invoice 注册为 function tool，由 LLM 原生 tool calling 返回结构化参数。
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.program.function_program import get_function_tool
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from pydantic import BaseModel, Field, field_validator

load_dotenv(".env")

INVOICE_PDF = Path("resources/invoice.pdf")

GENERIC_UNIT_LABELS = frozenset(
    {"项", "件", "个", "次", "月", "年", "服务费", "服务", "台", "套", "批"}
)

EXTRACTION_INSTRUCTION = """\
Extract invoice information from the following Chinese e-invoice text.

Rules:
- Each table row is ONE line item.
- item_name must come ONLY from the 项目名称 column. Merge broken lines.
- Values in the 单位 column (e.g. 服务费, 项) are NOT separate line items.
- price is the 金额 column value (pre-tax).
- invoice_id is the 发票号码 when present.

Invoice text:
{text}
"""


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


def extract_invoice_with_tool(llm: OpenAI, pdf_text: str) -> Invoice:
    tool = get_function_tool(Invoice)
    response = llm.chat_with_tools(
        [tool],
        user_msg=EXTRACTION_INSTRUCTION.format(text=pdf_text),
        tool_required=True,
    )

    tool_calls = llm.get_tool_calls_from_response(
        response,
        error_on_no_tool_call=True,
    )

    for tool_call in tool_calls:
        if tool_call.tool_name == "Invoice":
            invoice = Invoice(**tool_call.tool_kwargs)
            invoice.line_items = normalize_line_items(invoice.line_items)
            return fill_invoice_id(invoice, pdf_text)

    raise ValueError("No Invoice tool call returned from the LLM.")


def main() -> None:
    llm = configure_llm()
    pdf_path = Path(os.getenv("INVOICE_PDF_PATH", INVOICE_PDF))

    print(f"Reading PDF: {pdf_path}")
    pdf_text = load_pdf_text(pdf_path)

    print("Extracting via tool calling...\n")
    invoice = extract_invoice_with_tool(llm, pdf_text)

    print("--- Pydantic object ---")
    print(invoice)
    print("\n--- JSON ---")
    print(json.dumps(invoice.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
