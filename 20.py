"""
Structured LLM 示例：从 PDF 发票中提取结构化 Invoice 数据。

使用 Pydantic 定义 schema，通过 llm.as_structured_llm(Invoice) 将 PDF 文本
解析为带类型的 Invoice 对象。
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from pydantic import BaseModel, Field, field_validator

load_dotenv(".env")

INVOICE_PDF = Path("resources/invoice.pdf")

# 中国电子发票表格里常见“单位”列取值，不是独立商品名称
GENERIC_UNIT_LABELS = frozenset(
    {"项", "件", "个", "次", "月", "年", "服务费", "服务", "台", "套", "批"}
)

EXTRACTION_PROMPT = """\
Extract invoice information from the following Chinese e-invoice text.

Rules for line_items:
- Each table row under the columns [项目名称, 规格型号, 单位, 数量, 单价, 金额, 税率/征收率, 税额] is ONE line item.
- item_name must come ONLY from the 项目名称 column. Merge broken lines into a single name.
- Values in the 单位 column (e.g. 服务费, 项) are NOT separate line items.
- amount is the 金额 column value (pre-tax). price should equal amount for a single-quantity row.
- Do NOT duplicate the same row as multiple line items.
- invoice_id is the 发票号码 value when present.

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

    invoice_id: str = Field(
        description="发票号码。找不到时可留空字符串。"
    )
    date: datetime = Field(description="开票日期")
    line_items: list[LineItem] = Field(
        description="发票明细行，每一行表格记录对应一个元素。"
    )


def configure_llm() -> OpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ["OPENAI_BASE_URL"]
    model = os.environ["OPENAI_MODEL"]
    os.environ.setdefault("OPENAI_API_BASE", api_base)

    llm = OpenAI(model=model, api_key=api_key, api_base=api_base)
    Settings.llm = llm
    return llm


def load_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = PDFReader().load_data(file=pdf_path)
    if not documents:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    return documents[0].text


def normalize_line_items(line_items: list[LineItem]) -> list[LineItem]:
    """Remove unit-column labels mistaken as separate line items."""
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

    # Same amount duplicated: keep the most descriptive item name.
    by_amount: dict[float, LineItem] = {}
    for item in filtered:
        current = by_amount.get(item.price)
        if current is None or len(item.item_name) > len(current.item_name):
            by_amount[item.price] = item

    return list(by_amount.values())


def fill_invoice_id(invoice: Invoice, pdf_text: str) -> Invoice:
    if invoice.invoice_id and invoice.invoice_id not in {"未知", "N/A", "unknown"}:
        return invoice

    match = re.search(r"\b(\d{20})\b", pdf_text)
    if match:
        invoice.invoice_id = match.group(1)
    return invoice


def extract_invoice(llm: OpenAI, pdf_text: str) -> Invoice:
    sllm = llm.as_structured_llm(Invoice)
    response = sllm.complete(EXTRACTION_PROMPT.format(text=pdf_text))
    invoice = response.raw
    invoice.line_items = normalize_line_items(invoice.line_items)
    return fill_invoice_id(invoice, pdf_text)


def main() -> None:
    llm = configure_llm()
    pdf_path = Path(os.getenv("INVOICE_PDF_PATH", INVOICE_PDF))

    print(f"Reading PDF: {pdf_path}")
    pdf_text = load_pdf_text(pdf_path)

    print("Extracting structured invoice data...\n")
    invoice = extract_invoice(llm, pdf_text)

    print("--- Pydantic object ---")
    print(invoice)
    print("\n--- JSON ---")
    print(json.dumps(invoice.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
