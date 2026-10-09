# backend/core/pending.py
# 低置信问题 -> knowledge_pending 待补充队列（人工后续补充知识库）。
# 入队是旁路动作：任何失败只记日志，绝不影响主问答链路。

from backend.core.logger import get_logger
from backend.dependencies import get_mysql_client

logger = get_logger(__name__)


def enqueue_pending(question: str, answer: str = "", confidence: float = 0.0) -> None:
    """把低置信问答写入待补充队列（MySQL 不可用时静默跳过）。"""
    try:
        client = get_mysql_client()
        if not getattr(client, "available", False):
            logger.warning("pending.skip_mysql_down")
            return
        client.insert_pending(question=question, answer=answer, confidence=confidence)
    except Exception as e:
        logger.warning("pending.enqueue_failed", error=str(e))
