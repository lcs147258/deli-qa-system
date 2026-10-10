# backend/core/reranker.py
# 精排双后端 + retrieve 一体化流水线。
#   local       本地 bge-reranker-large（CrossEncoder，省 API、吃 ~1.2GB 内存）
#   siliconflow 在线 BAAI/bge-reranker-v2-m3（200~600ms、零内存，4GB 服务器默认；
#               超时/限流自动重试，彻底故障降级为 RRF 融合序，不阻塞问答）
# retrieve：子块双路召回 -> 父块回升（Small-to-Big）-> 父块精排。
# 注意：换 reranker 后端后分数尺度变化，high_confidence_threshold 必须用标准题重新标定。

import os
from dataclasses import dataclass
from typing import Optional

import httpx

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
RERANK_MAX_INPUT_CHARS = 1200   # 父块截断，防超模型 max_length
_MAX_RETRIES = 2


@dataclass
class RankedDocument:
    """精排后的单个文档结果。"""
    content: str
    score: float           # reranker 相关性分（两后端均为 [0, 1] 尺度）
    original_index: int
    metadata: dict


# ──────────────────────────────────────────────────────────────
# 精排后端
# ──────────────────────────────────────────────────────────────

class BaseReranker:
    def rerank_with_confidence(
        self, query: str, documents: list[dict], top_k: int = 3,
        min_score: float = 0.0,
    ) -> tuple[list[RankedDocument], float]:
        raise NotImplementedError

    @staticmethod
    def _assemble(query: str, documents: list[dict],
                  scores: list[float], top_k: int,
                  min_score: float) -> tuple[list[RankedDocument], float]:
        """统一后处理：排序 -> min_score 过滤 -> 截 top_k -> 取 top1 置信度。"""
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
            candidates=len(documents), kept=len(top_results),
            confidence=round(confidence, 4))
        return top_results, confidence


class LocalReranker(BaseReranker):
    """bge-reranker-large 本地 CrossEncoder（单例）。"""

    _instance: Optional["LocalReranker"] = None

    def __init__(self):
        import torch
        from sentence_transformers import CrossEncoder

        os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
        settings = get_settings()
        model_path = os.path.abspath(
            settings.path_from_root(settings.reranker_model_path))

        use_local = (
            os.path.exists(model_path)
            and os.path.isdir(model_path)
            and any(f.endswith((".bin", ".safetensors", ".json"))
                    for f in os.listdir(model_path))
        )
        model_id = model_path if use_local else "BAAI/bge-reranker-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.local_loading", model_id=model_id, device=device)
        self._model = CrossEncoder(model_id, device=device, max_length=512)
        logger.info("reranker.local_loaded", model_id=model_id)

    @classmethod
    def get_instance(cls) -> "LocalReranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank_with_confidence(self, query, documents, top_k=3, min_score=0.0):
        if not documents:
            return [], 0.0
        pairs = [
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]
        scores = self._model.predict(pairs).tolist()
        return self._assemble(query, documents, scores, top_k, min_score)


class SiliconFlowReranker(BaseReranker):
    """SiliconFlow /v1/rerank（BAAI/bge-reranker-v2-m3），失败由上层降级。"""

    _instance: Optional["SiliconFlowReranker"] = None

    def __init__(self):
        settings = get_settings()
        if not settings.sf_api_key:
            raise RuntimeError("RERANK_BACKEND=siliconflow 但未配置 SF_API_KEY")
        self._model = settings.rerank_sf_model
        self._client = httpx.Client(
            base_url=settings.sf_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.sf_api_key}"},
            timeout=settings.sf_timeout,
            trust_env=False,
        )
        logger.info("reranker.sf_init", model=self._model)

    @classmethod
    def get_instance(cls) -> "SiliconFlowReranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _request_scores(self, query: str, texts: list[str]) -> list[float]:
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.post(
                    "/rerank",
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": texts,
                        "top_n": len(texts),
                        "return_documents": False,
                    },
                )
                if resp.status_code in (429, 500, 502, 503) and attempt < _MAX_RETRIES:
                    logger.warning("reranker.sf_retry",
                                   status=resp.status_code, attempt=attempt + 1)
                    continue
                resp.raise_for_status()
                results = resp.json()["results"]
                scores = [0.0] * len(texts)
                for item in results:
                    scores[item["index"]] = float(item["relevance_score"])
                return scores
            except Exception as e:
                last_err = e
                if attempt < _MAX_RETRIES:
                    continue
        raise RuntimeError(f"SiliconFlow rerank 请求失败: {last_err}")

    def rerank_with_confidence(self, query, documents, top_k=3, min_score=0.0):
        if not documents:
            return [], 0.0
        texts = [(doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS]
                 for doc in documents]
        try:
            scores = self._request_scores(query, texts)
        except Exception as e:
            logger.error("reranker.sf_failed_degrade_rrf", error=str(e)[:200])
            return _rrf_fallback_rank(documents, top_k)
        return self._assemble(query, documents, scores, top_k, min_score)


def _rrf_fallback_rank(
    documents: list[dict], top_k: int,
) -> tuple[list[RankedDocument], float]:
    """reranker 在线后端彻底故障：按召回融合分 minmax 映射到 [0.5, 0.85] 序。

    刻意压在高置信阈值附近而非给满分：API 恢复前的回答保守放行，日志可监控。
    """
    n = len(documents)
    raw = [float(documents[i].get("score", 0.0)) for i in range(n)]
    order = sorted(range(n), key=lambda i: raw[i], reverse=True)
    lo, hi = (min(raw), max(raw)) if raw else (0.0, 0.0)
    ranked = []
    for new_pos, i in enumerate(order):
        norm = ((raw[i] - lo) / (hi - lo)) if hi - lo > 1e-9 else 1.0
        score = 0.5 + 0.35 * norm
        ranked.append(RankedDocument(
            content=documents[i].get("content", ""), score=score,
            original_index=i, metadata=documents[i].get("metadata", {})))
    top_results = ranked[:top_k]
    confidence = top_results[0].score if top_results else 0.0
    logger.warning("reranker.fallback_rrf_order",
                   candidates=n, confidence=round(confidence, 4))
    return top_results, confidence


_RERANKER: Optional[BaseReranker] = None


def get_reranker() -> BaseReranker:
    """按 RERANK_BACKEND 返回精排后端单例；在线端初始化失败（如缺 key）也降级本地。"""
    global _RERANKER
    if _RERANKER is None:
        backend = get_settings().rerank_backend.lower()
        if backend == "siliconflow":
            try:
                _RERANKER = SiliconFlowReranker.get_instance()
            except Exception as e:
                logger.error("reranker.sf_init_failed_use_local", error=str(e)[:200])
                _RERANKER = LocalReranker.get_instance()
        elif backend == "local":
            _RERANKER = LocalReranker.get_instance()
        else:
            raise ValueError(f"未知 RERANK_BACKEND: {backend}")
    return _RERANKER


# ──────────────────────────────────────────────────────────────
# 父子分块检索流水线
# ──────────────────────────────────────────────────────────────

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
      编码 -> 子块双路召回 -> 回升去重父块 -> 对父块精排。
    返回的 RankedDocument.content 是父块全文，调用方无需感知向量化。
    embedding 在线后端故障时降级纯 BM25 召回，不阻塞问答。
    """
    from backend.core.embeddings import EmbedderUnavailable, get_embedder
    from backend.core.knowledge_base import KnowledgeBaseClient

    settings = get_settings()
    recall_top_k = recall_top_k if recall_top_k is not None else settings.recall_top_k
    rerank_top_k = rerank_top_k if rerank_top_k is not None else settings.rerank_top_k
    min_score = min_score if min_score is not None else settings.min_relevant_score

    try:
        query_vec = get_embedder().encode_query(query)
    except EmbedderUnavailable as e:
        logger.warning("retrieve.embedder_failed_bm25_only", error=str(e)[:200])
        query_vec = None

    kb = KnowledgeBaseClient()
    child_hits = kb.hybrid_search(
        query_text=query,
        query_embedding=query_vec,
        top_k=recall_top_k,
        tenant_id=tenant_id,
        course_id=course_id,
    )
    if not child_hits:
        logger.info("retrieve.empty", query_preview=query[:50])
        return [], 0.0

    # 子块 -> 父块（同父块多子块去重，保留最高召回分）
    parent_candidates = kb.promote_to_parents(child_hits)

    return get_reranker().rerank_with_confidence(
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
    return get_reranker().rerank_with_confidence(
        query, parent_candidates, top_k=top_k, min_score=min_score)
