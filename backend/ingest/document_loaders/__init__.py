# document_loaders 包：迁移自 rag_qa/edu_document_loaders。
# OCR 加载器按扩展名在 loaders.py 中惰性导入，缺可选依赖不影响 txt/md 主链路。
from .edu_ocr import get_ocr

__all__ = ["get_ocr"]
