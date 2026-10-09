# scripts/init_milvus.py
# 重建 DeliRAG 知识库集合：
#   1) 备份旧库 deli.deli_final（标量字段导出 logs/deli_final_backup.json）
#   2) 删除旧集合（旧 IP+IVF_FLAT 父子块结构废弃）
#   3) 新建 knowledge_domain（HNSW+COSINE / SPARSE+IP / tenant+course INVERTED）
# 用法：
#   python scripts/init_milvus.py             # 交互确认
#   python scripts/init_milvus.py --force     # 跳过确认（流水线用）

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import DataType, MilvusClient

from backend.config import PROJECT_ROOT, get_settings

VECTOR_DIM = 1024
OLD_DB = "deli"
OLD_COLLECTION = "deli_final"
BACKUP_FILE = Path(PROJECT_ROOT) / "logs" / "deli_final_backup.json"


def backup_old_collection(client: MilvusClient) -> int:
    """把旧 deli.deli_final 的标量内容导出为 JSON。返回导出条数。"""
    legacy = MilvusClient(
        uri=f"http://{get_settings().milvus_host}:{get_settings().milvus_port}",
        db_name=OLD_DB,
    )
    if not legacy.has_collection(OLD_COLLECTION):
        print(f"ℹ️  旧集合 {OLD_DB}.{OLD_COLLECTION} 不存在，跳过备份")
        return 0

    rows = legacy.query(
        collection_name=OLD_COLLECTION,
        filter="id != ''",
        output_fields=["id", "text", "parent_id", "parent_content", "source", "timestamp"],
        limit=16384,
    )
    BACKUP_FILE.parent.mkdir(exist_ok=True)
    BACKUP_FILE.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"✅ 旧集合已备份：{BACKUP_FILE}（{len(rows)} 行）")
    return len(rows)


def build_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    # 父子分块：子块向量命中后按 parent_id 回升父块，parent_content 冗余父块全文
    schema.add_field("parent_id", DataType.VARCHAR, max_length=128)
    schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)
    schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("document_id", DataType.VARCHAR, max_length=64)
    schema.add_field("course_id", DataType.VARCHAR, max_length=64)
    schema.add_field("source_name", DataType.VARCHAR, max_length=256)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
    schema.add_field("version", DataType.VARCHAR, max_length=32)
    schema.add_field("updated_at", DataType.INT64)
    return schema


def build_index_params(client: MilvusClient):
    ip = client.prepare_index_params()
    ip.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 256})
    ip.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="IP", params={"drop_ratio_build": 0.2})
    ip.add_index(field_name="tenant_id", index_type="INVERTED")
    ip.add_index(field_name="course_id", index_type="INVERTED")
    return ip


def main(force: bool):
    settings = get_settings()
    uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    collection = settings.milvus_collection
    print(f"连接 Milvus：{uri}")

    if not force:
        ans = input(
            f"将备份并删除旧集合 {OLD_DB}.{OLD_COLLECTION}，重建 {collection}。\n"
            f"确认继续？输入 yes：").strip().lower()
        if ans != "yes":
            print("已取消")
            return

    client = MilvusClient(uri=uri)

    backup_old_collection(client)
    try:
        legacy = MilvusClient(uri=uri, db_name=OLD_DB)
        if legacy.has_collection(OLD_COLLECTION):
            print(f"🗑️  删除旧集合 {OLD_DB}.{OLD_COLLECTION} ...")
            legacy.drop_collection(OLD_COLLECTION)
    except Exception as e:
        print(f"⚠️  删除旧集合失败（可手动处理）：{e}")

    if client.has_collection(collection):
        print(f"🗑️  删除已存在的集合 '{collection}' ...")
        client.drop_collection(collection)

    client.create_collection(
        collection_name=collection,
        schema=build_schema(client),
        index_params=build_index_params(client),
    )
    print(f"✅ 集合 '{collection}' 创建完成（含索引，已加载）")
    print("当前集合：", client.list_collections())
    print("⚠️  请继续运行 build_knowledge_base.py 导入知识文档。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="跳过交互确认")
    main(force=parser.parse_args().force)
