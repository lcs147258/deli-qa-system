# backend/dependencies.py
# 依赖装配层：集中提供全局单例（连接复用），API 层 / Agent 层从这里取组件。
# 所有外部中间件的故障都在组件内部消化为"降级"，这里不再向上抛连接错误。

from backend.faq.faq_system import FAQSystem, get_faq_system


def get_faq() -> FAQSystem:
    """FAQ 通道单例（内含 MySQL/Redis 连接与 BM25 语料）。"""
    return get_faq_system()


def get_mysql_client():
    """MySQL 连接单例（复用 FAQ 通道的连接，供会话历史 / 待补充队列使用）。"""
    return get_faq_system().mysql_client
