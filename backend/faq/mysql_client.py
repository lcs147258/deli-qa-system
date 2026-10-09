# backend/faq/mysql_client.py
# MySQL 客户端（迁移自 mysql_qa/db/mysql_client.py）：
# faq 问答对 / conversations 会话历史 / knowledge_pending 待补充队列。
# 改造点：配置走 Settings；连接惰性建立且失败不阻断 Web 服务启动（FAQ 通道降级为不可用）。

import time
import uuid

import pymysql

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class MySQLClient:
    """封装连接、建表、FAQ 导入/查询、会话历史、待补充队列入库。"""

    RETRY_COOLDOWN = 10.0   # 连接失败后的重连冷却（秒），避免每个请求都卡 TCP 超时

    def __init__(self):
        self.settings = get_settings()
        self.connection = None
        self.cursor = None
        self.available = False
        self._last_failed_at = 0.0
        self._connect()

    def _connect(self) -> bool:
        """建立连接；失败返回 False（不抛异常，便于服务降级启动）。"""
        try:
            self.connection = pymysql.connect(**self.settings.mysql_dsn)
            self.cursor = self.connection.cursor()
            self.available = True
            logger.info("mysql.connected", database=self.settings.mysql_database)
            return True
        except pymysql.MySQLError as e:
            self.available = False
            self._last_failed_at = time.time()
            logger.warning("mysql.connect_failed", error=str(e))
            return False

    def _ensure_alive(self) -> bool:
        """执行前确认连接可用；失败冷却期内直接返回 False，不重复卡超时。"""
        if self.available:
            return True
        if time.time() - self._last_failed_at < self.RETRY_COOLDOWN:
            return False
        return self._connect()

    # ── 建表（幂等）──────────────────────────────────────────

    def ensure_tables(self) -> None:
        """创建本系统使用的三张表（均 IF NOT EXISTS，可重复执行）。"""
        if not self._ensure_alive():
            return
        try:
            with self.connection.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS faq (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        category VARCHAR(30),
                        question VARCHAR(500),
                        answer VARCHAR(2000)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(36) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        INDEX idx_session_id (session_id)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_pending (
                        id VARCHAR(64) PRIMARY KEY,
                        question VARCHAR(500) NOT NULL,
                        answer TEXT,
                        confidence FLOAT,
                        status VARCHAR(20) DEFAULT 'pending',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            self.connection.commit()
            logger.info("mysql.tables_ready")
        except pymysql.MySQLError as e:
            logger.error("mysql.ensure_tables_failed", error=str(e))

    def create_table(self):
        """兼容旧初始化脚本的方法名（仅建 faq 表）。"""
        if not self._ensure_alive():
            return
        with self.connection.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS faq (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    category VARCHAR(30),
                    question VARCHAR(500),
                    answer VARCHAR(2000)
                )
            """)
        self.connection.commit()
        logger.info("mysql.faq_table_ready")

    # ── FAQ 数据导入 ─────────────────────────────────────────

    def insert_data(self, csv_path):
        """从 CSV 批量导入问答对（表头：分类,问题,答案）。%s 占位防注入。"""
        import pandas as pd
        if not self._ensure_alive():
            raise RuntimeError("MySQL 不可用，无法导入 FAQ 数据")
        try:
            data = pd.read_csv(csv_path, encoding='utf-8-sig')
            with self.connection.cursor() as cur:
                for _, row in data.iterrows():
                    cur.execute(
                        "insert into faq(category, question, answer) values(%s, %s, %s)",
                        (row['分类'], row['问题'], row['答案']),
                    )
            self.connection.commit()
            logger.info("mysql.faq_imported", count=len(data))
        except Exception as e:
            self.connection.rollback()
            logger.error("mysql.faq_import_failed", error=str(e))
            raise

    def insert_missing(self, csv_path) -> int:
        """以 CSV 为事实源按 question 差异同步：插入新问题、更新答案有变化的旧问题。
        返回 (新增条数, 更新条数)。"""
        import pandas as pd
        if not self._ensure_alive():
            raise RuntimeError("MySQL 不可用，无法导入 FAQ 数据")
        try:
            data = pd.read_csv(csv_path, encoding='utf-8-sig')
            with self.connection.cursor() as cur:
                cur.execute("SELECT id, question, answer FROM faq")
                old = {q: (qid, ans) for qid, q, ans in cur.fetchall()}
                to_insert, to_update = [], []
                for _, r in data.iterrows():
                    cat, q, a = r['分类'], r['问题'], r['答案']
                    if q not in old:
                        to_insert.append((cat, q, a))
                    elif old[q][1] != a:
                        to_update.append((cat, a, old[q][0]))
                cur.executemany(
                    "insert into faq(category, question, answer) values(%s, %s, %s)",
                    to_insert)
                cur.executemany(
                    "update faq set category=%s, answer=%s where id=%s",
                    to_update)
            self.connection.commit()
            logger.info("mysql.faq_synced",
                        added=len(to_insert), updated=len(to_update))
            return len(to_insert), len(to_update)
        except Exception as e:
            self.connection.rollback()
            logger.error("mysql.faq_incremental_import_failed", error=str(e))
            raise

    # ── FAQ 查询（BM25 语料）─────────────────────────────────

    def fetch_questions(self):
        if not self._ensure_alive():
            return ()
        self.cursor.execute('select question from faq')
        return self.cursor.fetchall()

    def fetch_answer(self, question):
        if not self._ensure_alive():
            return None
        self.cursor.execute('select answer from faq where question=%s', (question,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def count(self):
        if not self._ensure_alive():
            return 0
        self.cursor.execute('select count(*) from faq')
        return self.cursor.fetchone()[0]

    # ── 会话历史（服务前端 /api/history，保留最近 5 轮）──────

    def fetch_recent_history(self, session_id, limit=5):
        """返回最近 limit 轮问答，按时间正序。"""
        if not self._ensure_alive():
            return []
        try:
            self.cursor.execute("""
                SELECT question, answer FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC LIMIT %s
            """, (session_id, limit))
            rows = self.cursor.fetchall()
            return [{"question": q, "answer": a} for q, a in rows][::-1]
        except pymysql.MySQLError as e:
            logger.error("mysql.fetch_history_failed", error=str(e))
            return []

    def insert_conversation(self, session_id, question, answer, keep_rounds=5):
        """插入一轮对话并只保留最近 keep_rounds 轮。失败静默回滚。"""
        if not self._ensure_alive():
            return
        try:
            self.cursor.execute(
                "INSERT INTO conversations (session_id, question, answer, timestamp) "
                "VALUES (%s, %s, %s, NOW())",
                (session_id, question, answer),
            )
            self.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
                  AND id NOT IN (
                      SELECT id FROM (
                          SELECT id FROM conversations
                          WHERE session_id = %s
                          ORDER BY timestamp DESC LIMIT %s
                      ) AS sub)
            """, (session_id, session_id, keep_rounds))
            self.connection.commit()
        except pymysql.MySQLError as e:
            logger.error("mysql.insert_conversation_failed", error=str(e))
            self.connection.rollback()

    def clear_history(self, session_id) -> bool:
        if not self._ensure_alive():
            return False
        try:
            self.cursor.execute(
                "DELETE FROM conversations WHERE session_id = %s", (session_id,))
            self.connection.commit()
            return True
        except pymysql.MySQLError as e:
            logger.error("mysql.clear_history_failed", error=str(e))
            self.connection.rollback()
            return False

    # ── 待补充队列（低置信问题人工补充知识库）────────────────

    def insert_pending(self, question: str, answer: str = "", confidence: float = 0.0):
        """低置信问答入队，幂等（同问题仅留一条）。失败静默。"""
        if not self._ensure_alive():
            return
        try:
            self.cursor.execute(
                "SELECT id FROM knowledge_pending WHERE question = %s", (question,))
            if self.cursor.fetchone():
                return
            self.cursor.execute("""
                INSERT INTO knowledge_pending (id, question, answer, confidence, status)
                VALUES (%s, %s, %s, %s, 'pending')
            """, (str(uuid.uuid4()), question, answer, confidence))
            self.connection.commit()
            logger.info("mysql.pending_inserted", question=question[:50])
        except pymysql.MySQLError as e:
            logger.warning("mysql.insert_pending_failed", error=str(e))
            try:
                self.connection.rollback()
            except Exception:
                pass

    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("mysql.closed")
        except pymysql.MySQLError as e:
            logger.error("mysql.close_failed", error=str(e))
