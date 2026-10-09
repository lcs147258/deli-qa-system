# backend/ingest 包：文档加载（含 OCR）与切分，建库脚本专用。
from backend.ingest.loaders import load_document, split_documents

__all__ = ["load_document", "split_documents"]
