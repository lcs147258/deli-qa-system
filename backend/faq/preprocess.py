# backend/faq/preprocess.py
# BM25 分词入口（迁移自 mysql_qa/utils/preprocess.py，逻辑不变）。
# BM25 基于"词"，中文必须先分词；熟食领域词要整词注册，否则"虎皮鸡爪"被切碎检索就废了。

import jieba

from backend.core.logger import get_logger

logger = get_logger(__name__)

# 熟食领域自定义词
_DELI_WORDS = [
    '卤味', '卤菜', '熟食', '凉菜', '酱鸭', '桶子鸡', '虎皮鸡爪', '猪头肉',
    '调味料汁', '八大拌', '五香料汁', '凉菜红油', '现拌', '效期', '报损',
    '冷链', '真空包装', '散装', '袋装', '批发', '加盟', '智慧出品',
]
for _w in _DELI_WORDS:
    jieba.add_word(_w)


def preprocess_text(text):
    """中文文本 -> 词列表（语料和查询同口径）。"""
    if not text or not isinstance(text, str):
        logger.warning("preprocess.invalid_text", text=repr(text))
        return []
    return jieba.lcut(text.lower())
