import operator
from typing import Annotated, TypedDict, List, Optional, Literal, Dict, Any
from pydantic import BaseModel
from docling_core.types.doc import DoclingDocument

class Evidence(BaseModel):
    page_no: Optional[int] = None
    quote: Optional[str] = None
    table_id: Optional[str] = None
    confidence: float = 0.0

class VerbatimMetric(BaseModel):
    raw_name: str
    raw_value: Optional[str] = None
    raw_unit: Optional[str] = None
    raw_context: Optional[str] = None
    evidence: Evidence

class NormalizedMetric(BaseModel):
    catalog_id: str
    value_num: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    stance: Literal["defense","attack","unknown"] = "unknown"
    domain: Literal["business","technical","cyber"] = "cyber"
    time_window: Optional[str] = None
    scope: Optional[str] = None
    evidence: Evidence

class GraphState(TypedDict, total=False):
    pdf_path: str
    doc_id: str
    doc_markdown: str
    tables: List[Dict[str, Any]]      # например: {table_id, page_no, md, csv_path}
    verbatim: List[VerbatimMetric]
    normalized: List[NormalizedMetric]
    report_md: str
    report_pdf_path: str


class GraphState_parallel(TypedDict, total=False):
    pdf_path: str
    doc_id: str
    doc_markdown: str
    docling_doc: DoclingDocument
    # map-reduce поля:
    pages: List[Dict[str, Any]]  # [{page_no: int, md: str}, ...]
    page_reports: Annotated[List[Dict[str, Any]], operator.add]  # reducer: append
    report_md: str