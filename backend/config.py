# backend/config.py
# 全项目唯一配置中心：从 .env.local 读取所有配置项（对齐 EduAgent 的 Settings 模式）。

import os
from functools import lru_cache

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# __file__ 位于 <项目根>/backend/config.py，向上两级即项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(PROJECT_ROOT, ".env.local")


class Settings(BaseSettings):
    """配置模型：每个属性对应 .env.local 的一项，Pydantic 自动做类型校验。"""

    # ── PostgreSQL / ParadeDB（第二版知识库：pgvector + pg_search 单库）──
    pg_host: str = "127.0.0.1"
    pg_port: int = 5434
    pg_user: str = "deli"
    pg_password: str = "delipg123"
    pg_database: str = "deli_qa"
    pg_table: str = "kb_chunks"
    default_tenant_id: str = "tenant_deli"

    # ── 大模型（DashScope OpenAI 兼容端点）──
    llm_api_key: str
    llm_base_url: str
    llm_model: str = "qwen3.7-plus"
    llm_temperature: float = 0
    llm_max_tokens: int = 2048

    # ── MySQL（FAQ / 会话历史 / 待补充队列）──
    mysql_host: str = "localhost"
    mysql_port: int = 3307
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "deli_qa"

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6380
    redis_password: str = "1234"
    redis_db: int = 0

    # ── 本地模型权重 ──
    # embedding 本地后端：bge-small-zh-v1.5（512 维 / 95MB，4GB 服务器可常驻 CPU）
    embedding_local_model_path: str = "E:/edurag_models/bge-small-zh-v1.5"
    reranker_model_path: str = "E:/edurag_models/bge-reranker-large"
    bert_base_model_path: str = "E:/edurag_models/bert-base-chinese"
    finetuned_classifier_path: str = "models/classifier/bert_query_classifier"

    # ── Embedding 双后端：local（本地小模型）| siliconflow（在线 bge-m3）──
    # 注意：两模型向量维度/空间不同，切换后端必须重建知识库（init_pg.py --force + 入库）。
    embedding_backend: str = "local"
    embedding_sf_model: str = "BAAI/bge-m3"
    embedding_dim: int = 512          # local=512；siliconflow bge-m3=1024
    embedding_batch_size: int = 32

    # ── Reranker 双后端：local（bge-reranker-large）| siliconflow（v2-m3 API）──
    # 服务器默认 siliconflow：零内存占用，中文效果持平或更优；API 故障自动降级 RRF 序。
    rerank_backend: str = "local"
    rerank_sf_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_batch_size: int = 32

    # ── SiliconFlow 在线接口（embedding/rerank 共用；免费额度 RPM 2000 起）──
    sf_api_key: str = ""
    sf_base_url: str = "https://api.siliconflow.cn/v1"
    sf_timeout: float = 15.0

    # ── 检索参数 ──
    faq_bm25_threshold: float = 0.85
    recall_top_k: int = 10
    rerank_top_k: int = 3
    min_relevant_score: float = 0.05
    high_confidence_threshold: float = 0.45
    # 双路召回：每路在 PG 内先取的候选池大小（RRF 融合后再截 recall_top_k）
    recall_candidate_limit: int = 50

    # ── 双路融合策略：rrf | weighted_rrf | combmnz（融合只定候选池，终答仍看 reranker）──
    fusion_strategy: str = "rrf"
    rrf_k: int = 60
    fusion_dense_weight: float = 0.6
    fusion_sparse_weight: float = 0.4
    # ── 父子分块（Parent-Child / Small-to-Big）──
    # 入库切子块（小粒度，向量精准命中），子块冗余携带 parent_content；
    # 检索时子块命中 -> 回升父块（大粒度，给 LLM 完整上下文）-> 父块 rerank。
    chunk_size: int = 512            # 兼容旧配置名（= 子块大小）
    parent_chunk_size: int = 1200
    child_chunk_size: int = 300
    chunk_overlap: int = 64
    contextual_rag: bool = True

    # ── 业务常量 ──
    valid_sources: list[str] = ["product", "safety", "ops", "franchise", "supply"]
    customer_service_phone: str = "13112345678"

    # ── 服务 ──
    app_host: str = "0.0.0.0"
    app_port: int = 8082
    log_level: str = "INFO"
    kb_mcp_server_port: int = 8001

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        # 本机同时跑教育版/熟食版，OS 环境变量存在同名项（如 LLM_API_KEY），
        # 必须让项目内 .env.local 优先，避免两个项目串配置。
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # ── 路径工具：相对路径一律解析为基于项目根的绝对路径 ──
    def path_from_root(self, relative_path: str) -> str:
        """把配置中的相对路径转成基于项目根的绝对路径；已是绝对路径则原样返回。"""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(PROJECT_ROOT, relative_path)

    @property
    def data_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, "data")

    @property
    def log_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, "logs")

    @property
    def classifier_abs_path(self) -> str:
        return self.path_from_root(self.finetuned_classifier_path)

    @property
    def mysql_dsn(self) -> dict:
        """pymysql 连接参数（不含库名的场景可自行 pop database）。"""
        return {
            "host": self.mysql_host,
            "port": self.mysql_port,
            "user": self.mysql_user,
            "password": self.mysql_password,
            "database": self.mysql_database,
            "charset": "utf8mb4",
        }

    @property
    def pg_dsn(self) -> str:
        """psycopg 连接串（ParadeDB / pgvector 知识库库）。"""
        return (
            f"host={self.pg_host} port={self.pg_port} "
            f"dbname={self.pg_database} user={self.pg_user} "
            f"password={self.pg_password} connect_timeout=5"
        )


@lru_cache()
def get_settings() -> Settings:
    """全局唯一配置对象。"""
    return Settings()
