# backend/core/knowledge_base.py
# 第二版知识库客户端：PostgreSQL（ParadeDB 镜像 = pgvector + pg_search 单库）。
#
# 双路召回 + 可插拔融合（backend/core/fusion.py）：
#   dense  路：pgvector HNSW + COSINE（embedding 由本地 bge-small-zh 或 SiliconFlow bge-m3 产出）
#   sparse 路：pg_search BM25，Lindera(中文) 分词建索引；
#              查询串在库内用同一分词器切原子词，boolean(should=>term[]) 做 bag-of-words，
#              避免默认查询串被当成"相邻短语"（"冷藏 柜 温度"场景会漏召回）。
#
# 接口与第一版 Milvus 客户端保持同名（promote_to_parents / upsert_chunks / count 等），
# 上层 reranker/入库脚本的数据契约不变：候选 = {content, score, metadata{...}}。

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from backend.config import get_settings
from backend.core.fusion import LaneHit, get_fusion
from backend.core.logger import get_logger

logger = get_logger(__name__)

_CHUNK_FIELDS = [
    "content", "source_name", "chunk_type", "course_id",
    "document_id", "chunk_index", "parent_id", "parent_content",
]
# Lindera 会把标点切成 token，查询侧只保留含汉字/字母/数字的词
_KEEP_TOKEN = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


class KnowledgeBaseClient:
    """PostgreSQL/ParadeDB 客户端单例：混合检索 + chunk 写入/删除。"""

    _conn: Optional["psycopg.Connection"] = None

    def __init__(self):
        if KnowledgeBaseClient._conn is None or KnowledgeBaseClient._conn.closed:
            settings = get_settings()
            KnowledgeBaseClient._conn = psycopg.connect(
                settings.pg_dsn, autocommit=True, row_factory=dict_row)
            logger.info("pg.connected",
                        host=settings.pg_host, port=settings.pg_port,
                        db=settings.pg_database)

    def _table(self) -> sql.Composed:
        return sql.Identifier(get_settings().pg_table)

    # ── 检索 ─────────────────────────────────────────────────

    def _lindera_tokens(self, query_text: str) -> list[str]:
        """查询串在库内走与 BM25 索引同一个 Lindera(中文) 分词器，保证词项对齐。"""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT %s::pdb.lindera(chinese)::text[] AS toks",
                    (query_text,))
                toks = cur.fetchone()["toks"] or []
            return [t for t in toks if t.strip() and _KEEP_TOKEN.search(t)]
        except Exception as e:
            logger.warning("knowledge_base.lindera_tokenize_failed",
                           error=str(e)[:200])
            return []

    def _dense_lane(self, query_embedding: list[float], tenant_id: str,
                    course_id: Optional[str], limit: int) -> list[LaneHit]:
        vec_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        query = sql.SQL(
            "SELECT id, 1 - (embedding <=> %s::vector) AS dscore FROM {tbl} "
            "WHERE tenant_id = %s {course} "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        ).format(
            tbl=self._table(),
            course=sql.SQL("AND course_id = %s") if course_id else sql.SQL(""),
        )
        params: list = [vec_str, tenant_id]
        if course_id:
            params.append(course_id)
        params += [vec_str, limit]
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return [LaneHit(id=r["id"], rank=i + 1, score=float(r["dscore"]))
                    for i, r in enumerate(cur.fetchall())]

    def _sparse_lane(self, tokens: list[str], tenant_id: str,
                     course_id: Optional[str], limit: int) -> list[LaneHit]:
        if not tokens:
            return []
        # 每个词项一个占位符，强制走 paradedb.term() 精确词项；should=OR 即 bag-of-words
        term_sql = sql.SQL(", ").join(
            sql.SQL("paradedb.term('content', {})").format(sql.Placeholder())
            for _ in tokens)
        query = sql.SQL(
            "SELECT id, paradedb.score(id) AS bscore FROM {tbl} "
            "WHERE content @@@ paradedb.boolean(should => ARRAY[{terms}]) "
            "AND tenant_id = %s {course} "
            "ORDER BY paradedb.score(id) DESC LIMIT %s"
        ).format(
            tbl=self._table(),
            terms=term_sql,
            course=sql.SQL("AND course_id = %s") if course_id else sql.SQL(""),
        )
        params: list = list(tokens) + [tenant_id]
        if course_id:
            params.append(course_id)
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return [LaneHit(id=r["id"], rank=i + 1, score=float(r["bscore"]))
                    for i, r in enumerate(cur.fetchall())]

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[list[float]],
        top_k: int,
        tenant_id: str,
        course_id: Optional[str] = None,
    ) -> list[dict]:
        """dense + sparse 双路召回 -> 可插拔融合 -> 回表取子块。

        query_embedding 为 None（在线 embedding 故障）时退化为纯 BM25；
        BM25 无有效词项时退化为纯向量；两者皆空返回 []。
        """
        try:
            limit = get_settings().recall_candidate_limit

            dense_hits = (
                self._dense_lane(query_embedding, tenant_id, course_id, limit)
                if query_embedding else [])
            tokens = self._lindera_tokens(query_text)
            sparse_hits = (
                self._sparse_lane(tokens, tenant_id, course_id, limit)
                if tokens else [])

            if not dense_hits and not sparse_hits:
                logger.info("knowledge_base.hybrid_search_empty",
                            query_preview=query_text[:50])
                return []

            fused = get_fusion().fuse(dense_hits, sparse_hits, top_k)
            ids = [cid for cid, _ in fused]
            score_map = dict(fused)

            fields = sql.SQL(", ").join(
                [sql.Identifier("id")] + list(map(sql.Identifier, _CHUNK_FIELDS)))
            with self._conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT {fields} FROM {tbl} WHERE id = ANY(%s)")
                    .format(fields=fields, tbl=self._table()),
                    (ids,))
                rows = {r["id"]: r for r in cur.fetchall()}

            candidates = []
            for cid in ids:
                entity = rows.get(cid)
                if not entity:
                    continue
                candidates.append({
                    "content": entity.get("content") or "",
                    "score": score_map[cid],
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
            logger.info("knowledge_base.hybrid_search_done",
                        dense=len(dense_hits), sparse=len(sparse_hits),
                        tokens=tokens, candidates=len(candidates))
            return candidates
        except Exception as e:
            logger.error("knowledge_base.hybrid_search_failed", error=str(e)[:300])
            return []

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
        """批量 upsert 文档子块（主键冲突则更新），单条多值 INSERT 原子提交。"""
        if not chunks:
            return 0
        cols = [
            "id", "content", "embedding", "parent_id", "parent_content",
            "chunk_index", "document_id", "course_id", "source_name",
            "chunk_type", "version", "tenant_id", "updated_at",
        ]
        query = sql.SQL(
            "INSERT INTO {tbl} ({fields}) VALUES ({placeholders}) "
            "ON CONFLICT (id) DO UPDATE SET "
            "content = EXCLUDED.content, embedding = EXCLUDED.embedding, "
            "parent_id = EXCLUDED.parent_id, "
            "parent_content = EXCLUDED.parent_content, "
            "chunk_index = EXCLUDED.chunk_index, document_id = EXCLUDED.document_id, "
            "course_id = EXCLUDED.course_id, source_name = EXCLUDED.source_name, "
            "chunk_type = EXCLUDED.chunk_type, version = EXCLUDED.version, "
            "tenant_id = EXCLUDED.tenant_id, updated_at = EXCLUDED.updated_at"
        ).format(
            tbl=self._table(),
            fields=sql.SQL(", ").join(map(sql.Identifier, cols)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
        )
        rows = [(
            c.id, c.content,
            "[" + ",".join(str(float(x)) for x in c.embedding) + "]",
            c.parent_id[:128], c.parent_content or c.content,
            c.chunk_index, c.document_id, c.course_id, c.source_name,
            c.chunk_type, c.version, c.tenant_id, c.updated_at,
        ) for c in chunks]
        with self._conn.cursor() as cur:
            cur.executemany(query, rows, returning=False)
        logger.info("knowledge_base.chunks_upserted", count=len(chunks))
        return len(chunks)

    def delete_document_chunks(self, document_id: str) -> None:
        """删除指定文档的全部 chunk（重建时先删后插，幂等）。"""
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {tbl} WHERE document_id = %s")
                .format(tbl=self._table()),
                (document_id,))
        logger.info("knowledge_base.document_deleted", document_id=document_id)

    def count(self, tenant_id: Optional[str] = None) -> int:
        """知识库当前行数（可按租户过滤）。"""
        try:
            with self._conn.cursor() as cur:
                if tenant_id:
                    cur.execute(
                        sql.SQL("SELECT count(*) FROM {tbl} WHERE tenant_id = %s")
                        .format(tbl=self._table()), (tenant_id,))
                else:
                    cur.execute(
                        sql.SQL("SELECT count(*) FROM {tbl}").format(tbl=self._table()))
                return int(cur.fetchone()["count"])
        except Exception as e:
            logger.warning("knowledge_base.count_failed", error=str(e))
            return -1

    @staticmethod
    def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
        """内容+位置不变则 ID 不变，支持幂等 upsert。"""
        raw = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class DocumentChunk:
    """写入 kb_chunks 的单个子块，字段与表结构一一对应（父子分块）。"""
    id: str
    content: str                       # 子块文本（参与向量编码/BM25 召回）
    embedding: list[float]
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
