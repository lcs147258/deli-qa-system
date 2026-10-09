# backend/ingest/loaders.py
# 文档加载与切分统一入口（建库脚本调用）：
#   load_document(path)      按扩展名路由：txt/md 直读，pdf/docx/ppt/图片走 OCR 加载器
#   split_documents(docs)    md 先按标题切，再统一递归字符切分（默认 512/64）
# OCR 依赖较重且为可选链路，统一惰性导入：缺依赖时只有用到对应类型才报错。

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".doc",
    ".ppt", ".pptx", ".jpg", ".jpeg", ".png",
}


def load_document(file_path: str) -> list:
    """加载单个文件为 langchain Document 列表。txt/md 强制 utf-8。"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()

    if ext == ".pdf":
        from backend.ingest.document_loaders.edu_pdfloader import OCRPDFLoader
        return OCRPDFLoader(file_path).load()

    if ext in (".docx", ".doc"):
        from backend.ingest.document_loaders.edu_docloader import OCRDOCLoader
        return OCRDOCLoader(file_path).load()

    if ext in (".ppt", ".pptx"):
        from backend.ingest.document_loaders.edu_pptloader import OCRPPTLoader
        return OCRPPTLoader(file_path).load()

    if ext in (".jpg", ".jpeg", ".png"):
        from backend.ingest.document_loaders.edu_imgloader import OCRIMGLoader
        return OCRIMGLoader(file_path).load()

    raise ValueError(f"不支持的文件类型: {file_path}")


def _split_markdown(text: str, recursive_splitter) -> list:
    """Markdown 先按 #~#### 标题切（保留标题行在正文里），再递归细切。"""
    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    try:
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False,
        )
        return md_splitter.split_text(text)
    except Exception as e:
        logger.warning("ingest.markdown_split_failed", error=str(e))
        return recursive_splitter.create_documents([text])


def split_documents(documents: list, parent_size: int = None,
                    child_size: int = None, chunk_overlap: int = None) -> list:
    """
    父子两层切分（Parent-Child / Small-to-Big）：
      父块 parent_size（默认 1200，保留完整上下文）
        └─ 子块 child_size（默认 300，向量精准命中），冗余携带 parent_content
    .md 先按 #~#### 标题切出章节段，章节段内再做父子切分。
    返回子块 Document 列表，每个子块 metadata 含：
      parent_id / parent_content / id（子块全局ID）
    """
    from langchain_core.documents import Document

    settings = get_settings()
    parent_size = parent_size or settings.parent_chunk_size
    child_size = child_size or settings.child_chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=chunk_overlap, length_function=len)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=chunk_overlap, length_function=len)

    child_chunks: list = []
    for doc in documents:
        source_path = (doc.metadata or {}).get("source") \
            or (doc.metadata or {}).get("file_path") or ""
        ext = os.path.splitext(source_path)[1].lower()
        doc_uid = (doc.metadata or {}).get("doc_uid") or "doc"

        # 1) 先得到"章节段"（md 按标题，其余整篇即一段）
        if ext == ".md":
            sections = _split_markdown(doc.page_content, parent_splitter)
        else:
            sections = [doc]

        parent_index = 0
        for section in sections:
            section_meta = {**(doc.metadata or {}), **(getattr(section, "metadata", {}) or {})}
            # 2) 章节段 -> 父块
            parent_docs = parent_splitter.create_documents(
                [section.page_content], metadatas=[section_meta])

            for parent_doc in parent_docs:
                parent_id = f"{doc_uid}_p{parent_index}"
                parent_text = parent_doc.page_content
                parent_index += 1

                # 3) 父块 -> 子块，子块冗余携带父块全文
                sub_texts = child_splitter.split_text(parent_text)
                for child_index, sub_text in enumerate(sub_texts):
                    child_meta = {
                        **parent_doc.metadata,
                        "parent_id": parent_id,
                        "parent_content": parent_text,
                        "id": f"{parent_id}_c{child_index}",
                    }
                    child_chunks.append(
                        Document(page_content=sub_text, metadata=child_meta))

    logger.info("ingest.documents_split_parent_child",
                documents=len(documents), child_chunks=len(child_chunks),
                parent_size=parent_size, child_size=child_size,
                overlap=chunk_overlap)
    return child_chunks
