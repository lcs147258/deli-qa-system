# backend/faq/faq_system.py
# FAQ 通道门面：组合 MySQL + Redis + BM25，对外暴露 answer()。
# QA 图的 faq_check 节点调用它；组件任一不可用时 answer() 直接返回 None 走 RAG。

from functools import lru_cache

from backend.core.logger import get_logger
from backend.faq.bm25_search import BM25Search
from backend.faq.mysql_client import MySQLClient
from backend.faq.redis_client import RedisClient

logger = get_logger(__name__)


class FAQSystem:
    def __init__(self):
        self.mysql_client = MySQLClient()
        # 建表幂等执行；MySQL 不可用时内部自行降级
        self.mysql_client.ensure_tables()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    @property
    def available(self) -> bool:
        return self.bm25_search.available

    def answer(self, query: str, threshold: float = None):
        """命中 FAQ 返回答案字符串，未命中返回 None。"""
        if not self.available:
            return None
        answer, need_rag = self.bm25_search.search(query, threshold=threshold)
        if need_rag:
            return None
        return answer


@lru_cache()
def get_faq_system() -> FAQSystem:
    """全局单例（含连接复用）。"""
    return FAQSystem()
