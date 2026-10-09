# backend/mcp/knowledge_base_server.py
# DeliRAG 知识库 MCP 工具服务（Streamable HTTP）：
# 对外暴露 search_knowledge_base，内部走 BGE-M3 混合检索 + BGE reranker 精排。
# 启动：python -m backend.mcp.knowledge_base_server  （默认 8001）

import asyncio
import os
import pathlib
import sys

# 允许「python backend/mcp/knowledge_base_server.py」直接运行
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(str(_PROJECT_ROOT / ".env.local"))

mcp = FastMCP(
    name="DeliRAG-KnowledgeBase",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def search_knowledge_base(
    query: str,
    tenant_id: str = "tenant_deli",
    course_id: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    在 DeliRAG 熟食企业知识库中做 Hybrid（语义+关键词）检索并精排。

    Args:
        query:     检索问题（中文）
        tenant_id: 租户 ID，默认 tenant_deli
        course_id: 知识分类过滤，可选，取值
                   product(产品规格) / safety(食品安全) / ops(门店运营)
                   / franchise(加盟) / supply(供应链)
        top_k:     精排后返回条数（默认 3）

    Returns:
        命中块列表：content / source_name / score / confidence / is_high_confidence
    """
    from backend.config import get_settings
    from backend.core.logger import get_logger
    from backend.core.reranker import retrieve

    logger = get_logger(__name__)
    settings = get_settings()

    if course_id is not None and course_id not in settings.valid_sources:
        raise ValueError(
            f"非法 course_id: {course_id}，可选：{settings.valid_sources}")

    # 编码/精排是同步阻塞推理，放线程池
    loop = asyncio.get_running_loop()
    ranked_docs, confidence = await loop.run_in_executor(
        None,
        lambda: retrieve(
            query=query,
            tenant_id=tenant_id,
            course_id=course_id,
            rerank_top_k=top_k,
        ),
    )

    logger.info("kb_mcp.search_done",
                query_preview=query[:50], hits=len(ranked_docs),
                confidence=round(confidence, 4))

    threshold = settings.high_confidence_threshold
    return [
        {
            "content": doc.content,
            "source_name": doc.metadata.get("source_name", ""),
            "score": round(doc.score, 6),
            "confidence": round(confidence, 4),
            "is_high_confidence": confidence >= threshold,
        }
        for doc in ranked_docs
    ]


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("KB_MCP_SERVER_PORT", "8001"))
    print(f"DeliRAG Knowledge Base MCP Server -> http://0.0.0.0:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)
