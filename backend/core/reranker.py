# backend/core/reranker.py
# BGE reranker-large 精排（CrossEncoder 单例）+ retrieve 一体化 Pipeline。
# 迁移自 EduAgent；deli 改造：权重 bge-reranker-large、阈值走 Settings，
# 低于 min_relevant_score 的候选视为 ANN 兜底噪声直接丢弃。

import os
from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
RERANK_MAX_INPUT_CHARS = 1200   # 防超出 CrossEncoder max_length=512


@dataclass
class RankedDocument:
    """精排后的单个文档结果。"""
    content: str
    score: float           # BGE-Reranker sigmoid 相关性分 [0, 1]
    original_index: int
    metadata: dict


class BGEReranker:
    _instance: Optional["BGEReranker"] = None

    def __init__(self):
        os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
        settings = get_settings()
        model_path = os.path.abspath(settings.path_from_root(settings.reranker_model_path))

        use_local = (
            os.path.exists(model_path)
            and os.path.isdir(model_path)
            and any(f.endswith((".bin", ".safetensors", ".json"))
                    for f in os.listdir(model_path))
        )
        model_id = model_path if use_local else "BAAI/bge-reranker-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.loading", model_id=model_id, device=device)
        self._model = CrossEncoder(model_id, device=device, max_length=512)
        logger.info("reranker.loaded", model_id=model_id)

    @classmethod
    def get_instance(cls) -> "BGEReranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank_with_confidence(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> tuple[list[RankedDocument], float]:
        """
        精排返回 (top 文档, top1 置信度)。
        min_score 以下的文档视为无关召回，直接过滤（分数不参与 confidence）。
        """
        if not documents:
            return [], 0.0

        pairs = [
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted([
            RankedDocument(
                content=documents[i].get("content", ""),
                score=float(scores[i]),
                original_index=i,
                metadata=documents[i].get("metadata", {}),
            )
            for i in range(len(documents))
        ], key=lambda x: x.score, reverse=True)

        if min_score > 0:
            ranked = [d for d in ranked if d.score >= min_score]

        top_results = ranked[:top_k]
        confidence = top_results[0].score if top_results else 0.0

        logger.info(
            "reranker.done",
            candidates=len(documents),
            kept=len(top_results),
            confidence=round(confidence, 4),
        )
        return top_results, confidence


def retrieve(
    query: str,
    tenant_id: str,
    course_id: Optional[str] = None,
    recall_top_k: int = None,
    rerank_top_k: int = None,
    min_score: float = None,
) -> tuple[list[RankedDocument], float]:
    """
    父子分块检索（Small-to-Big）：
      编码 -> 子块 Hybrid 召回 -> 回升去重父块 -> 对父块精排。
    返回的 RankedDocument.content 是父块全文，调用方无需感知向量化。
    """
    from backend.core.knowledge_base import BGEMEmbedder, KnowledgeBaseClient  # 延迟导入防循环

    settings = get_settings()
    recall_top_k = recall_top_k if recall_top_k is not None else settings.recall_top_k
    rerank_top_k = rerank_top_k if rerank_top_k is not None else settings.rerank_top_k
    min_score = min_score if min_score is not None else settings.min_relevant_score

    embedder = BGEMEmbedder.get_instance()
    dense_vec, sparse_vec = embedder.encode_query(query)

    kb = KnowledgeBaseClient()
    filters = kb._build_filter(tenant_id, course_id)
    child_hits = kb._hybrid_search(
        query_embedding=dense_vec,
        query_sparse=sparse_vec,
        top_k=recall_top_k,
        filters=filters,
    )
    if not child_hits:
        logger.info("retrieve.empty", query_preview=query[:50])
        return [], 0.0

    # 子块 -> 父块（同父块多子块去重，保留最高召回分）
    parent_candidates = kb.promote_to_parents(child_hits)

    reranker = BGEReranker.get_instance()
    return reranker.rerank_with_confidence(
        query, parent_candidates, top_k=rerank_top_k, min_score=min_score)


def rerank_parents(
    query: str,
    parent_candidates: list[dict],
    top_k: int = None,
    min_score: float = None,
) -> tuple[list[RankedDocument], float]:
    """对多路（如子查询）合并后的父块候选做一次统一精排。"""
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.rerank_top_k
    min_score = min_score if min_score is not None else settings.min_relevant_score
    if not parent_candidates:
        return [], 0.0
    return BGEReranker.get_instance().rerank_with_confidence(
        query, parent_candidates, top_k=top_k, min_score=min_score)
