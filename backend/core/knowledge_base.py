# backend/core/knowledge_base.py
# Milvus 知识库客户端 + BGE-M3 编码器（单集合 knowledge_domain，tenant 隔离）。
# 迁移自 EduAgent 同构实现：HNSW + COSINE dense / SPARSE_INVERTED_INDEX + IP sparse，
# WeightedRanker(0.7, 0.3) 融合。deli 改造：集合名/默认租户/模型路径走 Settings。

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from backend.config import PROJECT_ROOT, get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseClient:
    """Milvus 客户端单例：混合检索 + chunk 写入/删除。"""

    ANN_EF = 64   # HNSW 搜索候选集大小（精度/速度平衡点）

    _client: Optional["MilvusClient"] = None
    _loaded: bool = False

    def __init__(self):
        if KnowledgeBaseClient._client is None:
            settings = get_settings()
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            KnowledgeBaseClient._client = MilvusClient(uri=uri)
            logger.info("milvus.connected", uri=uri)

        if not KnowledgeBaseClient._loaded:
            try:
                KnowledgeBaseClient._client.load_collection(get_settings().milvus_collection)
            except Exception:
                pass   # init_milvus.py 尚未运行时忽略
            KnowledgeBaseClient._loaded = True

    # ── 检索 ─────────────────────────────────────────────────

    def _hybrid_search(
        self,
        query_embedding: list[float],
        query_sparse: dict,
        top_k: int,
        filters: Optional[str] = None,
    ) -> list[dict]:
        """Dense + Sparse 双路 AnnSearchRequest -> WeightedRanker 融合。"""
        try:
            dense_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": self.ANN_EF}},
                limit=top_k,
                expr=filters,
            )
            sparse_req = AnnSearchRequest(
                data=[query_sparse],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filters,
            )

            output_fields = [
                "content", "source_name", "chunk_type",
                "course_id", "document_id", "chunk_index",
                "parent_id", "parent_content",
            ]

            results = self._client.hybrid_search(
                collection_name=get_settings().milvus_collection,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )

            candidates = []
            for hit in results[0]:
                entity = hit["entity"]
                candidates.append({
                    "content": entity.get("content") or "",
                    "score": hit.get("distance") or 0.0,
                    "metadata": {
                        "source_name": entity.get("source_name") or "",
                        "chunk_type": entity.get("chunk_type") or "text",
                        "course_id": entity.get("course_id") or "",
                        "document_id": entity.get("document_id") or "",
                        "chunk_index": entity.get("chunk_index") or 0,
                        "parent_id": entity.get("parent_id") or "",
                        "parent_content": entity.get("parent_content") or "",
                    },
                })

            logger.info("knowledge_base.hybrid_search_done", candidates=len(candidates))
            return candidates

        except Exception as e:
            logger.error("knowledge_base.hybrid_search_failed", error=str(e))
            return []

    @staticmethod
    def _build_filter(tenant_id: str, course_id: Optional[str] = None) -> str:
        """拼 Milvus bool 表达式；对 tenant_id / course_id 转义防注入。"""
        safe_tenant = tenant_id.replace('"', '\\"')
        expr = f'tenant_id == "{safe_tenant}"'
        if course_id:
            safe_course = course_id.replace('"', '\\"')
            expr += f' and course_id == "{safe_course}"'
        return expr

    # ── 父子块回升（Small-to-Big）────────────────────────────

    @staticmethod
    def promote_to_parents(child_candidates: list[dict]) -> list[dict]:
        """
        子块召回结果 -> 去重回升的父块列表。
        同一父块的多个子块只保留召回分数最高的一个；
        content 替换为父块全文（送给 reranker/LLM），子块元数据保留用于溯源。
        无 parent_content 的旧数据退化为子块自身。
        """
        merged: dict[str, dict] = {}
        order: list[str] = []
        for cand in child_candidates:
            meta = cand.get("metadata", {})
            parent_id = meta.get("parent_id") or ""
            parent_text = meta.get("parent_content") or cand.get("content") or ""
            key = parent_id or f"cid_{hashlib.md5(parent_text.encode()).hexdigest()[:16]}"
            if key not in merged:
                merged[key] = {
                    "content": parent_text,
                    "score": cand.get("score", 0.0),
                    "metadata": {
                        "source_name": meta.get("source_name", ""),
                        "course_id": meta.get("course_id", ""),
                        "document_id": meta.get("document_id", ""),
                        "parent_id": parent_id,
                    },
                }
                order.append(key)
            elif cand.get("score", 0.0) > merged[key]["score"]:
                merged[key]["score"] = cand["score"]
        parents = [merged[k] for k in order]
        logger.info("knowledge_base.parent_promoted",
                    children=len(child_candidates), parents=len(parents))
        return parents

    # ── 写入 ─────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list) -> int:
        """批量 upsert 文档块（主键存在则更新）。"""
        if not chunks:
            return 0
        data = [{
            "id": c.id,
            "embedding": c.embedding,
            "sparse_embedding": c.sparse_embedding,
            "content": c.content[:4096],
            "parent_id": c.parent_id[:128],
            "parent_content": (c.parent_content or c.content)[:65000],
            "chunk_index": c.chunk_index,
            "document_id": c.document_id,
            "course_id": c.course_id,
            "tenant_id": c.tenant_id,
            "source_name": c.source_name,
            "chunk_type": c.chunk_type,
            "version": c.version,
            "updated_at": c.updated_at,
        } for c in chunks]

        self._client.upsert(collection_name=get_settings().milvus_collection, data=data)
        logger.info("knowledge_base.chunks_upserted", count=len(chunks))
        return len(chunks)

    def delete_document_chunks(self, document_id: str) -> None:
        """删除指定文档的全部 chunk（重建时先删后插，幂等）。"""
        safe_id = document_id.replace('"', '\\"')
        self._client.delete(
            collection_name=get_settings().milvus_collection,
            filter=f'document_id == "{safe_id}"',
        )
        logger.info("knowledge_base.document_deleted", document_id=document_id)

    def count(self, tenant_id: Optional[str] = None) -> int:
        """集合当前行数（实时 query；get_collection_stats 对 growing segment 有延迟）。"""
        try:
            expr = (f'tenant_id == "{tenant_id}"' if tenant_id
                    else 'tenant_id != ""')
            rows = self._client.query(
                collection_name=get_settings().milvus_collection,
                filter=expr, output_fields=["id"], limit=16384)
            return len(rows)
        except Exception as e:
            logger.warning("knowledge_base.count_failed", error=str(e))
            return -1

    @staticmethod
    def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
        """内容+位置不变则 ID 不变，支持幂等 upsert。"""
        raw = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────
# BGE-M3 编码器（dense 1024 维 + sparse 词权，进程内单例）
# ──────────────────────────────────────────────────────────────

class BGEMEmbedder:
    _instance: Optional["BGEMEmbedder"] = None

    def __init__(self, model_path: str):
        # 兼容补丁1：FlagEmbedding 1.3.x 引用 transformers 内部函数，保险兜底
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu
        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = (
                lambda: _ilu.find_spec("torch.fx") is not None
            )

        # 兼容补丁2：部分 transformers 版本 XLMRobertaModel 不收 dtype 参数，
        # FlagEmbedding 会传入，子类化丢弃它
        from transformers.models.xlm_roberta import modeling_xlm_roberta as _xlm
        _OriginalXLMRoberta = _xlm.XLMRobertaModel

        class _PatchedXLMRobertaModel(_OriginalXLMRoberta):
            def __init__(self, config, **kwargs):
                kwargs.pop("dtype", None)
                super().__init__(config, **kwargs)

        _xlm.XLMRobertaModel = _PatchedXLMRobertaModel

        import torch
        from FlagEmbedding import BGEM3FlagModel

        # BGE 权重必须走纯 ASCII 路径（sentencepiece C++ 层窄字符 API 读不了中文路径）
        logger.info("bge_m3.loading", model_path=model_path)
        _use_fp16 = torch.cuda.is_available()
        self._model = BGEM3FlagModel(
            model_name_or_path=model_path,
            use_fp16=_use_fp16,
        )
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        if cls._instance is None:
            settings = get_settings()
            model_path = settings.path_from_root(settings.bge_m3_model_path)
            cls._instance = BGEMEmbedder(os.path.abspath(model_path))
        return cls._instance

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """批量编码 -> (dense 向量列表, sparse {token_id: weight} 列表)。"""
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = output["dense_vecs"].tolist()
        # numpy.float16 -> Python float：Milvus 返回 numpy 标量，统一转原生 float
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]
        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """单条查询编码（batch_size=1）。"""
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]


@dataclass
class DocumentChunk:
    """写入 Milvus 的单个子块，字段与集合 Schema 一一对应（父子分块）。"""
    id: str
    content: str                       # 子块文本（参与向量编码/召回）
    embedding: list[float]
    sparse_embedding: dict
    course_id: str          # deli 五分类：product/safety/ops/franchise/supply
    document_id: str
    source_name: str
    chunk_type: str
    chunk_index: int
    version: str
    parent_id: str = ""                 # 所属父块 ID
    parent_content: str = ""            # 父块全文（冗余，检索时 Small-to-Big 回升）
    tenant_id: str = "tenant_deli"
    updated_at: int = field(default_factory=lambda: int(time.time()))


def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()
