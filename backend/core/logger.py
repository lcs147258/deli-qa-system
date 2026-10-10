# backend/core/logger.py
# 全项目日志工具：标准库 logging 之上支持「事件名 + 键值对」结构化写法
# （对齐 EduAgent），同时保留 DeliRAG 原有的 控制台 + logs/app.log 双输出。

import logging
import os
import sys

from backend.config import PROJECT_ROOT, get_settings


class _Logger:
    """让 logger.info("事件名", key=value) 与 logger.info("模板 %s", 值) 两种写法都可用。"""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def _fmt(self, event: str, **kw) -> str:
        if kw:
            return event + " | " + " ".join(f"{k}={v!r}" for k, v in kw.items())
        return event

    def debug(self, event: str, *args, **kw):
        if args:
            self._log.debug(event, *args)
        else:
            self._log.debug(self._fmt(event, **kw))

    def info(self, event: str, *args, **kw):
        if args:
            self._log.info(event, *args)
        else:
            self._log.info(self._fmt(event, **kw))

    def warning(self, event: str, *args, **kw):
        if args:
            self._log.warning(event, *args)
        else:
            self._log.warning(self._fmt(event, **kw))

    def error(self, event: str, *args, **kw):
        if args:
            self._log.error(event, *args)
        else:
            exc_info = kw.pop("exc_info", False)
            self._log.error(self._fmt(event, **kw), exc_info=exc_info)

    def critical(self, event: str, *args, **kw):
        if args:
            self._log.critical(event, *args)
        else:
            self._log.critical(self._fmt(event, **kw))


def configure_logging() -> None:
    """全局日志配置：应用启动时调用一次。控制台 + 文件双输出，压低三方库噪音。"""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    os.makedirs(settings.log_dir, exist_ok=True)
    log_file = os.path.join(settings.log_dir, "app.log")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    # force 清空旧 handler，避免 reload / 重复初始化时同一条日志打多遍
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers",
                  "transformers", "psycopg"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> _Logger:
    """模块统一入口：get_logger(__name__)。"""
    return _Logger(name)
