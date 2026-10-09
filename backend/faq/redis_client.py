# backend/faq/redis_client.py
# Redis 客户端（迁移自 mysql_qa/cache/redis_client.py）：
# Cache-Aside + Lazy Loading。与教育版共用 Redis 实例，deli: 前缀隔离。

import json

import redis

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    PREFIX = 'deli'   # 与教育版的 qa_* / answer:* 键隔离，防止语料缓存互相覆盖

    @classmethod
    def answer_key(cls, query):
        return f'{cls.PREFIX}:answer:{query}'

    def __init__(self):
        settings = get_settings()
        self.available = False
        try:
            self.client = redis.StrictRedis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            self.client.ping()
            self.available = True
            logger.info("redis.connected", port=settings.redis_port)
        except redis.RedisError as e:
            # 不抛异常：FAQ 通道在 Redis 不可用时退化为直查 MySQL
            self.client = None
            logger.warning("redis.connect_failed", error=str(e))

    def set_data(self, key, value):
        if not self.available:
            return
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False))
        except redis.RedisError as e:
            logger.error("redis.set_failed", key=key, error=str(e))

    def get_data(self, key):
        if not self.available:
            return None
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            logger.error("redis.get_failed", key=key, error=str(e))
            return None

    def get_answer(self, query):
        if not self.available:
            return None
        try:
            answer = self.client.get(self.answer_key(query))
            if answer:
                return json.loads(answer)   # 与 set_data 的 json.dumps 对称
            return None
        except redis.RedisError as e:
            logger.error("redis.get_answer_failed", error=str(e))
            return None
