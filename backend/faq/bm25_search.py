# backend/faq/bm25_search.py
# BM25 热路径（迁移自 mysql_qa/retrieval/bm25_search.py），逻辑不变：
# 答案缓存 -> softmax 归一化打分 -> >=阈值 命中 FAQ，否则交 RAG 通道。

import numpy as np
from rank_bm25 import BM25Okapi

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.faq.preprocess import preprocess_text

logger = get_logger(__name__)


class BM25Search:
    def __init__(self, redis_client, mysql_client, threshold: float = None):
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.threshold = threshold or get_settings().faq_bm25_threshold

        self.bm25 = None
        self.questions = None            # 分词后语料
        self.original_questions = None   # 原始问题（回查答案用）

        self._load_data()

    def _load_data(self):
        """Redis 缓存优先，未命中则 MySQL -> 分词 -> 回填。"""
        if not getattr(self.mysql_client, "available", False):
            logger.warning("faq.mysql_unavailable_channel_down")
            return

        original_key = 'deli:qa_original_questions'
        tokenized_key = 'deli:qa_tokenized_questions'

        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        if not self.original_questions or not tokenized_questions:
            rows = self.mysql_client.fetch_questions()
            if not rows:
                logger.warning("faq.empty_corpus")
                return
            tokenized_questions = [preprocess_text(q[0]) for q in rows]
            self.original_questions = [q[0] for q in rows]
            self.redis_client.set_data(original_key, self.original_questions)
            self.redis_client.set_data(tokenized_key, tokenized_questions)
            logger.info("faq.corpus_loaded_from_mysql", count=len(rows))

        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        logger.info("faq.bm25_ready", corpus=len(self.questions))

    @property
    def available(self) -> bool:
        return self.bm25 is not None

    @staticmethod
    def _softmax(scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def search(self, query, threshold=None):
        """
        :return: (answer, need_rag)
                 命中: (答案, False)；未命中/异常: (None, True) —— 降级走 RAG
        """
        threshold = threshold or self.threshold
        if not query or not isinstance(query, str):
            return None, True
        if not self.available:
            return None, True

        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer, False

        try:
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_score = self._softmax(scores)
            best_idx = int(softmax_score.argmax())
            best_score = float(softmax_score[best_idx])

            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    self.redis_client.set_data(
                        self.redis_client.answer_key(query), answer)
                    logger.info("faq.hit", score=round(best_score, 3),
                                question=original_question)
                    return answer, False

            logger.info("faq.miss", best_score=round(best_score, 3),
                        threshold=threshold)
            return None, True
        except Exception as e:
            logger.error("faq.search_error", error=str(e))
            return None, True
