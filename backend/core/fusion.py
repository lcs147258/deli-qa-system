# backend/core/fusion.py
# 双路召回融合策略（dense 向量路 + sparse BM25 路 -> 统一候选排序）。
#
# 关键认知：融合只决定"哪些父块进精排候选池"，最终 top3 与置信度由 cross-encoder
# reranker 决定，因此融合算法对终答影响有限。默认 RRF（无需分数尺度对齐、最稳）。
#
# 可配（.env.local FUSION_STRATEGY）：
#   rrf           1/(k+rank_d) + 1/(k+rank_s)，经典逆名次融合，对分数尺度免疫
#   weighted_rrf  两路分别乘 dense/sparse 权重的 RRF
#   combmnz       两路 minmax 归一后 CombSUM × 共同命中路数（偏好两路共识候选）

from dataclasses import dataclass

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LaneHit:
    """单路召回的一个候选：id、名次（从 1 开始）、该路原始分。"""
    id: str
    rank: int
    score: float


def _minmax(hits: list[LaneHit]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return {h.id: 1.0 for h in hits}
    return {h.id: (h.score - lo) / (hi - lo) for h in hits}


class FusionStrategy:
    name = "base"

    def fuse(self, dense: list[LaneHit], sparse: list[LaneHit],
             top_k: int) -> list[tuple[str, float]]:
        raise NotImplementedError


class RRFFusion(FusionStrategy):
    name = "rrf"

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, dense, sparse, top_k):
        scores: dict[str, float] = {}
        for h in dense:
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (self.k + h.rank)
        for h in sparse:
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (self.k + h.rank)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


class WeightedRRFFusion(FusionStrategy):
    name = "weighted_rrf"

    def __init__(self, k: int = 60, w_dense: float = 0.6, w_sparse: float = 0.4):
        self.k = k
        self.w_dense = w_dense
        self.w_sparse = w_sparse

    def fuse(self, dense, sparse, top_k):
        scores: dict[str, float] = {}
        for h in dense:
            scores[h.id] = scores.get(h.id, 0.0) + self.w_dense / (self.k + h.rank)
        for h in sparse:
            scores[h.id] = scores.get(h.id, 0.0) + self.w_sparse / (self.k + h.rank)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


class CombMNZFusion(FusionStrategy):
    """CombMNZ：minmax 归一分数求和，再乘"命中路数"（1 或 2）。"""

    name = "combmnz"

    def fuse(self, dense, sparse, top_k):
        nd, ns = _minmax(dense), _minmax(sparse)
        ids = set(nd) | set(ns)
        scores = {}
        for cid in ids:
            in_d, in_s = cid in nd, cid in ns
            scores[cid] = (nd.get(cid, 0.0) + ns.get(cid, 0.0)) * (in_d + in_s)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def get_fusion() -> FusionStrategy:
    s = get_settings()
    name = s.fusion_strategy.lower()
    if name == "rrf":
        return RRFFusion(k=s.rrf_k)
    if name == "weighted_rrf":
        return WeightedRRFFusion(
            k=s.rrf_k, w_dense=s.fusion_dense_weight,
            w_sparse=s.fusion_sparse_weight)
    if name == "combmnz":
        return CombMNZFusion()
    logger.warning("fusion.unknown_strategy_fallback_rrf", strategy=name)
    return RRFFusion(k=s.rrf_k)
