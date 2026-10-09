# scripts/init_faq_data.py
# 初始化 MySQL：建库 deli_qa + 建三张表（faq/conversations/knowledge_pending）
# + 增量导入 data/faq/deli_faq.csv（只插 MySQL 中缺失的 question）。可重复执行。
# 注意：新增 FAQ 后需清 Redis 语料缓存（deli:qa_original_questions /
#       deli:qa_tokenized_questions）并重启 Web 服务，BM25 才会重建。
# 用法：python scripts/init_faq_data.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pymysql

from backend.config import get_settings
from backend.faq.mysql_client import MySQLClient


def main():
    settings = get_settings()
    csv_path = str(Path(settings.data_dir) / "faq" / "deli_faq.csv")

    # 1) 不带库名连接，建库（CREATE DATABASE 不能参数化）
    conn = pymysql.connect(
        host=settings.mysql_host, port=settings.mysql_port,
        user=settings.mysql_user, password=settings.mysql_password,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS {settings.mysql_database} "
                f"CHARACTER SET utf8mb4")
        conn.commit()
        print(f"✅ 数据库 {settings.mysql_database} 就绪")
    finally:
        conn.close()

    # 2) 建表 + 幂等导入
    client = MySQLClient()
    try:
        client.ensure_tables()

        # 差异同步：CSV 为事实源，插入新 question、更新答案变化的旧 question
        before = client.count()
        added, updated = client.insert_missing(csv_path)
        if added or updated:
            print(f"✅ FAQ 同步：新增 {added} 条、更新 {updated} 条"
                  f"（{before} → {client.count()}），记得清 Redis 语料缓存并重启服务")
        else:
            print(f"ℹ️  faq 表 {before} 条与 CSV 完全一致，无变更")

        total = client.count()
        client.cursor.execute("SELECT question, answer FROM faq LIMIT 2")
        samples = client.cursor.fetchall()
        print(f"\nfaq 表共 {total} 条，抽样：")
        for q, a in samples:
            print(f"  问: {q}\n  答: {a[:40]}...")

        client.cursor.execute("SHOW TABLES")
        print("当前表：", [r[0] for r in client.cursor.fetchall()])
    finally:
        client.close()


if __name__ == "__main__":
    main()
