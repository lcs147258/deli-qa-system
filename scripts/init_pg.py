# scripts/init_pg.py
# 第二版知识库初始化（PostgreSQL / ParadeDB，替换旧 init_milvus.py）：
#   1) 启用扩展 vector（pgvector）+ pg_search（ParadeDB BM25）
#   2) 重建 kb_chunks 表：embedding vector(EMBEDDING_DIM)（HNSW+COSINE）、
#      父子分块字段、tenant/course 隔离字段
#   3) 建 pg_search BM25 索引：content 走 Lindera 中文分词
# 用法：
#   python scripts/init_pg.py             # 交互确认
#   python scripts/init_pg.py --force     # 跳过确认（流水线用）
#
# 注意：embedding 维度由 .env.local 的 EMBEDDING_DIM 决定
#       （local bge-small-zh=512 / siliconflow bge-m3=1024），切换 embedding 后端
#       必须先改配置，再跑本脚本 --force，然后重新 build_knowledge_base.py。

import argparse
import re
import sys
from pathlib import Path

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings   # noqa: E402

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def _ident(name: str) -> sql.Identifier:
    if not _IDENT.match(name):
        raise ValueError(f"非法 SQL 标识符：{name}")
    return sql.Identifier(name)


def main(force: bool):
    s = get_settings()
    table = s.pg_table
    print(f"连接 PostgreSQL：{s.pg_host}:{s.pg_port}/{s.pg_database}")
    print(f"知识库表：{table} | embedding 维度：{s.embedding_dim}")

    if not force:
        ans = input(f"将 DROP 并重建表 {table}（含全部索引）。\n确认继续？输入 yes：") \
            .strip().lower()
        if ans != "yes":
            print("已取消")
            return

    with psycopg.connect(s.pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
            cur.execute(
                "SELECT extname, extversion FROM pg_extension "
                "WHERE extname IN ('vector', 'pg_search') ORDER BY extname")
            for row in cur.fetchall():
                print(f"  扩展就绪：{row[0]} {row[1]}")

            # CASCADE：连同旧 BM25 索引一起删除
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(_ident(table)))

            cur.execute(sql.SQL(
                """
                CREATE TABLE {t} (
                    id             VARCHAR(64) PRIMARY KEY,
                    content        TEXT NOT NULL,
                    embedding      vector({dim}) NOT NULL,
                    parent_id      VARCHAR(128) NOT NULL DEFAULT '',
                    parent_content TEXT NOT NULL DEFAULT '',
                    tenant_id      VARCHAR(64) NOT NULL,
                    course_id      VARCHAR(64) NOT NULL DEFAULT '',
                    document_id    VARCHAR(64) NOT NULL,
                    source_name    VARCHAR(256) NOT NULL DEFAULT '',
                    chunk_type     VARCHAR(32) NOT NULL DEFAULT 'text',
                    chunk_index    BIGINT NOT NULL DEFAULT 0,
                    version        VARCHAR(32) NOT NULL DEFAULT '1.0',
                    updated_at     BIGINT NOT NULL DEFAULT 0
                )
                """
            ).format(t=_ident(table), dim=s.embedding_dim))

            # dense 路：HNSW + cosine（与旧 Milvus 度量一致）
            cur.execute(sql.SQL(
                "CREATE INDEX {idx} ON {t} USING hnsw "
                "(embedding vector_cosine_ops) WITH (m=16, ef_construction=256)"
            ).format(idx=_ident(f"idx_{table}_hnsw"), t=_ident(table)))

            # 标量过滤/删除用 btree
            for col in ("tenant_id", "course_id", "document_id", "parent_id"):
                cur.execute(sql.SQL(
                    "CREATE INDEX {idx} ON {t} ({col})"
                ).format(idx=_ident(f"idx_{table}_{col}"),
                         t=_ident(table), col=_ident(col)))

            # sparse 路：pg_search BM25，content 走 Lindera 中文分词
            cur.execute(sql.SQL(
                """
                CREATE INDEX {idx} ON {t}
                USING paradedb (id, (content::pdb.lindera(chinese)))
                WITH (key_field='id')
                """
            ).format(idx=_ident(f"idx_{table}_bm25"), t=_ident(table)))

            cur.execute(
                sql.SQL("SELECT count(*) FROM {}").format(_ident(table)))
            rows = cur.fetchone()[0]

    print(f"✅ 表 {table} 创建完成（HNSW + Lindera BM25），当前 {rows} 行")
    print("⚠️  请继续运行 build_knowledge_base.py 导入知识文档。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="跳过交互确认")
    main(force=parser.parse_args().force)
