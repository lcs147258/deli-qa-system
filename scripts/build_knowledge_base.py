# scripts/build_knowledge_base.py
# DeliRAG 知识库批量入库：
#   扫描 data/{course}_data/ 五类目录 -> 加载（含 OCR）-> 父子切分（父1200/子300）
#   -> 可选 Contextual RAG -> BGE-M3 双向量编码子块（冗余携带父块全文）-> 先删后写 Milvus。
#   检索时子块命中 -> 父块回升（Small-to-Big）-> 父块 rerank。
# 物理隔离：唯一扫描根是 data/，docs/（访谈提记、内部配方）永不入库。
#
# 用法：
#   python scripts/build_knowledge_base.py --no-context          # 全量快速入库
#   python scripts/build_knowledge_base.py --course product      # 只入产品类
#   python scripts/build_knowledge_base.py --file product_data/x.md --no-context

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.core.knowledge_base import (
    BGEMEmbedder,
    DocumentChunk,
    KnowledgeBaseClient,
)
from backend.core.llm_factory import get_llm
from backend.ingest import load_document, split_documents

MAX_CONTEXT_CONCURRENCY = 5

CONTEXTUAL_CHUNK_PROMPT = """\
<document>
{document_text}
</document>

以下是需要在整个文档中定位的片段：
<chunk>
{chunk_content}
</chunk>

请用一句简洁的中文，描述这段内容在整个熟食业务文档中的位置和主题，以便改善检索效果。
只输出这一句描述，不要加任何前缀或标签。"""


def stable_document_id(rel_path: str) -> str:
    """相对 data/ 的路径做 md5：同一路径永远得到同一 document_id，重建幂等。"""
    return hashlib.md5(rel_path.replace("\\", "/").encode("utf-8")).hexdigest()


def discover_files(data_dir: Path, courses: list[str]) -> list[tuple[Path, str]]:
    """返回 [(文件绝对路径, course_id)]；强校验路径必须位于 data/ 内。"""
    found = []
    allowed = {f"{c}_data" for c in courses}
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir() or sub.name not in allowed:
            continue
        course_id = sub.name[:-5]   # product_data -> product
        for path in sorted(sub.rglob("*")):
            if path.is_file() and path.suffix.lower() in (
                    ".txt", ".md", ".pdf", ".docx", ".doc",
                    ".ppt", ".pptx", ".jpg", ".jpeg", ".png"):
                resolved = path.resolve()
                if data_dir.resolve() not in resolved.parents:
                    raise RuntimeError(f"非法路径（不在 data/ 内）：{resolved}")
                if "docs" in resolved.parts:
                    raise RuntimeError(f"安全拦截：docs/ 目录禁止入库：{resolved}")
                found.append((resolved, course_id))
    return found


async def add_context(chunks, docs, concurrency=MAX_CONTEXT_CONCURRENCY):
    full_doc_text = "\n\n".join(d.page_content for d in docs)[:8000]
    llm = get_llm("qa", temperature=0)
    sem = asyncio.Semaphore(concurrency)

    async def gen(chunk):
        async with sem:
            try:
                text = await llm.achat([{
                    "role": "user",
                    "content": CONTEXTUAL_CHUNK_PROMPT.format(
                        document_text=full_doc_text,
                        chunk_content=chunk.page_content,
                    ),
                }])
                return text.strip()
            except Exception as e:
                print(f"   [warning] 上下文生成失败，保留原文：{e}")
                return ""

    contexts = await asyncio.gather(*[gen(c) for c in chunks])
    enriched = 0
    for chunk, ctx in zip(chunks, contexts):
        if ctx:
            chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
            enriched += 1
    print(f"  上下文增强：{enriched}/{len(chunks)} 个 chunk")
    return chunks


def embed_and_build(chunks, course_id, document_id, tenant_id):
    embedder = BGEMEmbedder.get_instance()
    texts = [c.page_content for c in chunks]
    dense_vecs, sparse_vecs = embedder.encode(texts, batch_size=12)

    built = []
    for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        meta = chunk.metadata or {}
        headers = [meta[k] for k in ("h1", "h2", "h3", "h4") if meta.get(k)]
        if headers:
            source_name = " > ".join(headers)
        else:
            source_name = Path(meta.get("source", "")).name or document_id[:8]

        chunk_id = KnowledgeBaseClient.generate_chunk_id(
            chunk.page_content, document_id, i)
        built.append(DocumentChunk(
            id=chunk_id,
            content=chunk.page_content,
            embedding=dense,
            sparse_embedding=sparse,
            course_id=course_id,
            document_id=document_id,
            source_name=source_name[:256],
            chunk_type="text",
            chunk_index=i,
            version="1.0",
            parent_id=meta.get("parent_id", ""),
            parent_content=meta.get("parent_content", chunk.page_content),
            tenant_id=tenant_id,
        ))
    return built


async def process_file(file_path: Path, course_id: str, data_dir: Path,
                       tenant_id: str, use_context: bool, kb: KnowledgeBaseClient):
    rel_path = str(file_path.relative_to(data_dir))
    document_id = stable_document_id(rel_path)
    print(f"\n{'='*60}\n📄 {rel_path}  → course={course_id} doc={document_id[:12]}")

    docs = load_document(str(file_path))
    # 注入文档级 UID，供父子切分生成稳定 parent_id（doc_uid_p{n}_c{k}）
    doc_uid = document_id[:16]
    for d in docs:
        d.metadata = {**(d.metadata or {}), "doc_uid": doc_uid}
    print(f"  读取：{len(docs)} 个文档，{sum(len(d.page_content) for d in docs)} 字")
    chunks = split_documents(docs)
    parent_n = len({c.metadata.get("parent_id") for c in chunks})
    print(f"  父子切分：{parent_n} 个父块 / {len(chunks)} 个子块")

    if use_context and chunks:
        chunks = await add_context(chunks, docs)

    built = embed_and_build(chunks, course_id, document_id, tenant_id)
    kb.delete_document_chunks(document_id)
    written = kb.upsert_chunks(built)
    return written


async def main(args):
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data 目录不存在：{data_dir}")

    tenant_id = args.tenant or settings.default_tenant_id
    courses = settings.valid_sources if args.course == "all" else [args.course]

    if args.file:
        file_path = (data_dir / args.file).resolve()
        if data_dir.resolve() not in file_path.parents:
            raise SystemExit("安全拦截：--file 必须位于 data/ 目录内")
        course_id = file_path.parent.name.removesuffix("_data")
        if course_id not in settings.valid_sources:
            raise SystemExit(f"无法从目录名识别 course：{file_path.parent.name}")
        targets = [(file_path, course_id)]
    else:
        targets = discover_files(data_dir, courses)

    if not targets:
        raise SystemExit("未发现可入库文件（检查 data/{course}_data/ 目录）")

    print(f"待入库 {len(targets)} 个文件 | tenant={tenant_id} | "
          f"Contextual RAG={'开' if not args.no_context else '关'}")
    kb = KnowledgeBaseClient()

    stats = {}
    total = 0
    for file_path, course_id in targets:
        n = await process_file(
            file_path, course_id, data_dir, tenant_id,
            use_context=not args.no_context, kb=kb)
        stats[course_id] = stats.get(course_id, 0) + n
        total += n

    print(f"\n{'='*60}\n🎉 入库完成：共 {total} 个 chunk")
    for c, n in stats.items():
        print(f"  {c}: {n}")
    print(f"集合总行数：{kb.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", default="all",
                        help="product/safety/ops/franchise/supply/all")
    parser.add_argument("--file", default=None, help="只处理 data/ 下指定文件")
    parser.add_argument("--no-context", action="store_true", help="跳过 Contextual RAG")
    parser.add_argument("--tenant", default=None)
    asyncio.run(main(parser.parse_args()))
