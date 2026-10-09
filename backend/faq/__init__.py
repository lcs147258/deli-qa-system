# backend/faq 包：FAQ 通道（MySQL 问答对 + Redis 缓存 + BM25 热路径命中）
from backend.faq.faq_system import FAQSystem, get_faq_system

__all__ = ["FAQSystem", "get_faq_system"]
