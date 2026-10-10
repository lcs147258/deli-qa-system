# backend/core/embeddings.py
# Embedding 双后端（第二版轻量化）：
#   local       本地 bge-small-zh-v1.5（512 维 / 95MB，CPU 可常驻，4GB 服务器默认）
#   siliconflow 在线 BAAI/bge-m3（1024 维，零内存；内容哈希缓存 + 批量 + 超时重试）
#
# 调用方只依赖统一接口：
#   embedder.dim                          向量维度（建表/校验用）
#   embedder.encode_documents(texts)      入库批量编码（失败必须抛异常，禁止静默缺向量）
#   embedder.encode_query(text)           查询编码（失败抛 EmbedderUnavailable，上层降级纯 BM25）
#
# 注意：两个后端向量空间不同，切换后端必须重建知识库（init_pg.py --force + 重新入库）。

import hashlib
import json
import os
import threading
from typing import Optional

import httpx

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

_MEM_CACHE_MAX = 2048          # 进程内哈希缓存上限（82 块语料 + 高频问足够）
_SF_MAX_BATCH = 32             # SiliconFlow embeddings 单批上限保守值
_MAX_RETRIES = 2


class EmbedderUnavailable(RuntimeError):
    """在线 embedding 后端连续失败（超时/限流/鉴权），上层据此降级纯 BM25。"""


class _HashedCache:
    """两层内容哈希缓存：进程内 dict + Redis（不可用时静默降级为仅内存）。"""

    def __init__(self, namespace: str, ttl: int = 30 * 24 * 3600):
        self._ns = namespace
        self._ttl = ttl
        self._mem: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._redis = None
        self._redis_tried = False

    def _key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"deli:embed:{self._ns}:{h}"

    def _get_redis(self):
        if self._redis_tried:
            return self._redis
        self._redis_tried = True
        try:
            import redis as redis_lib
            from backend.config import get_settings as _gs
            s = _gs()
            client = redis_lib.Redis(
                host=s.redis_host, port=s.redis_port, password=s.redis_password,
                db=s.redis_db, decode_responses=True, socket_connect_timeout=2,
            )
            client.ping()
            self._redis = client
        except Exception as e:
            logger.warning("embeddings.cache_redis_unavailable", error=str(e)[:120])
            self._redis = None
        return self._redis

    def get_many(self, texts: list[str]) -> dict[str, list[float]]:
        hit: dict[str, list[float]] = {}
        keys = {t: self._key(t) for t in texts}
        with self._lock:
            for t, k in keys.items():
                if t not in hit and k in self._mem:
                    hit[t] = self._mem[k]
        missing = [t for t in texts if t not in hit]
        if missing:
            r = self._get_redis()
            if r is not None:
                try:
                    vals = r.mget([keys[t] for t in missing])
                    for t, v in zip(missing, vals):
                        if v:
                            vec = json.loads(v)
                            hit[t] = vec
                            with self._lock:
                                self._mem[keys[t]] = vec
                except Exception as e:
                    logger.warning("embeddings.cache_read_failed", error=str(e)[:120])
        return hit

    def set_many(self, items: dict[str, list[float]]) -> None:
        if not items:
            return
        with self._lock:
            for t, vec in items.items():
                k = self._key(t)
                self._mem[k] = vec
            if len(self._mem) > _MEM_CACHE_MAX * 1.5:
                for old_k in list(self._mem)[:len(self._mem) - _MEM_CACHE_MAX]:
                    self._mem.pop(old_k, None)
        r = self._get_redis()
        if r is not None:
            try:
                pipe = r.pipeline()
                for t, vec in items.items():
                    pipe.set(self._key(t), json.dumps(vec), ex=self._ttl)
                pipe.execute()
            except Exception as e:
                logger.warning("embeddings.cache_write_failed", error=str(e)[:120])


class BaseEmbedder:
    dim: int = 0

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def encode_query(self, text: str) -> list[float]:
        return self.encode_documents([text])[0]


class LocalBGEEmbedder(BaseEmbedder):
    """本地 bge-small-zh-v1.5：sentence-transformers + CPU，输出 L2 归一化向量。"""

    def __init__(self):
        settings = get_settings()
        model_path = os.path.abspath(
            settings.path_from_root(settings.embedding_local_model_path))
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"本地 embedding 模型目录不存在：{model_path}")

        from sentence_transformers import SentenceTransformer
        logger.info("embedding.local_loading", model_path=model_path)
        self._model = SentenceTransformer(model_path, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._cache = _HashedCache("local-smallzh")
        logger.info("embedding.local_loaded", dim=self.dim)

    def _encode_uncached(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)
        return [v.astype(float).tolist() for v in vecs]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cached = self._cache.get_many(texts)
        miss = [t for t in texts if t not in cached]
        if miss:
            newly = dict(zip(miss, self._encode_uncached(miss)))
            self._cache.set_many(newly)
            cached.update(newly)
        return [cached[t] for t in texts]


class SiliconFlowEmbedder(BaseEmbedder):
    """SiliconFlow /v1/embeddings（默认 BAAI/bge-m3，1024 维）。"""

    def __init__(self):
        settings = get_settings()
        if not settings.sf_api_key:
            raise EmbedderUnavailable(
                "EMBEDDING_BACKEND=siliconflow 但未配置 SF_API_KEY")
        self._model = settings.embedding_sf_model
        self.dim = settings.embedding_dim
        self._timeout = settings.sf_timeout
        self._client = httpx.Client(
            base_url=settings.sf_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.sf_api_key}"},
            timeout=self._timeout,
            trust_env=False,
        )
        self._cache = _HashedCache("sf-bgem3")
        logger.info("embedding.sf_init", model=self._model, dim=self.dim)

    def _request_batch(self, batch: list[str]) -> list[list[float]]:
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.post(
                    "/embeddings",
                    json={"model": self._model, "input": batch,
                          "encoding_format": "float"},
                )
                if resp.status_code in (429, 500, 502, 503) and attempt < _MAX_RETRIES:
                    logger.warning("embedding.sf_retry",
                                   status=resp.status_code, attempt=attempt + 1)
                    continue
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]
            except Exception as e:
                last_err = e
                if attempt < _MAX_RETRIES:
                    continue
        raise EmbedderUnavailable(f"SiliconFlow embedding 请求失败: {last_err}")

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cached = self._cache.get_many(texts)
        miss = [t for t in texts if t not in cached]
        for i in range(0, len(miss), _SF_MAX_BATCH):
            batch = miss[i:i + _SF_MAX_BATCH]
            vecs = self._request_batch(batch)
            self._cache.set_many(dict(zip(batch, vecs)))
            cached.update(dict(zip(batch, vecs)))
        return [cached[t] for t in texts]


_EMBEDDER: Optional[BaseEmbedder] = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> BaseEmbedder:
    """按 EMBEDDING_BACKEND 返回进程级单例（local | siliconflow）。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                backend = get_settings().embedding_backend.lower()
                if backend == "siliconflow":
                    _EMBEDDER = SiliconFlowEmbedder()
                elif backend == "local":
                    _EMBEDDER = LocalBGEEmbedder()
                else:
                    raise ValueError(f"未知 EMBEDDING_BACKEND: {backend}")
    return _EMBEDDER
