# -*- coding: utf-8 -*-
"""
DeliRAG 一键启动脚本

用法：
  双击本文件，或在终端/VSCode 里运行
      python 启动服务.py

它会自动：
  1. 检查端口是否被占用（占用就提示，不会盲目启动）
  2. 检查中间件（ParadeDB/MySQL/Redis），没启动就帮你启动
  3. 启动 Web 服务

停止方式：
  · 在终端里按 Ctrl+C
  · 或双击「停止服务.py」
  · 或直接关闭窗口

注意：本脚本用【标准库】写成，任何 Python 都能跑（哪怕不是项目环境）。
      真正启动服务时会显式调用项目虚拟环境里的 Python。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

# ─────────────────────── 配置 ───────────────────────
PROJECT_DIR = Path(__file__).resolve().parent          # 本脚本所在目录 = 项目根目录
VENV_PYTHON = Path(r"E:\venvs\edurag\Scripts\python.exe")
# 第二版中间件容器（edurag-*，由 docker-compose.yml 定义）
#   paradedb = pgvector + pg_search 单容器（替代旧版 etcd/minio/milvus 三件套）
PORT = 8082
HOST = "127.0.0.1"
WAIT_AFTER_MIDDLEWARE = 20      # 启动中间件后等待的秒数（PG 就绪比 Milvus 快）
MIDDLEWARE_CONTAINERS = [
    "edurag-paradedb", "edurag-redis", "edurag-mysql",
]


def line(char="=", n=58):
    print(char * n)


def port_in_use(port: int) -> bool:
    """检测端口是否已被监听"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def middleware_running() -> bool:
    """检查 DeliRAG 使用的中间件容器是否在运行

    注意：容器名是 ASCII，但 Windows 上子进程输出可能是 GBK，
          所以统一用 errors='replace' 兜底，避免解码异常。
    """
    try:
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, timeout=15)
        out = r.stdout.decode("utf-8", errors="replace")
        return "edurag-paradedb" in out and "edurag-mysql" in out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_middleware() -> bool:
    """拉起 edurag-* 中间件容器。

    容器已由 docker compose 创建过（同名），直接 docker start 最可靠，
    避免在另一个 compose 项目里 up 触发"容器名冲突"。
    """
    try:
        r = subprocess.run(["docker", "start", *MIDDLEWARE_CONTAINERS],
                           capture_output=True, timeout=120)
        if r.returncode != 0:
            # 容器尚不存在（首次运行）-> 退回 compose 创建
            if (PROJECT_DIR / "docker-compose.yml").is_file():
                subprocess.run(["docker", "compose", "up", "-d"],
                               cwd=PROJECT_DIR, timeout=300)
            else:
                return False
        return True
    except FileNotFoundError:
        print("[错误] 找不到 docker 命令 —— 请确认 Docker Desktop 已安装并启动")
        return False
    except subprocess.TimeoutExpired:
        print("[错误] 启动 Docker 容器超时")
        return False


def main() -> int:
    line()
    print("   DeliRAG 熟食问答系统 - 启动")
    line()
    print()

    # ── 0. 检查 venv 是否存在 ──
    if not VENV_PYTHON.is_file():
        print(f"[错误] 找不到项目虚拟环境的 Python：")
        print(f"       {VENV_PYTHON}")
        print("       请确认 E:\\venvs\\edurag 是否还在。")
        return 1

    # ── 1. 端口检查 ──
    if port_in_use(PORT):
        print(f"[警告] 端口 {PORT} 已被占用！")
        print()
        print("       可能服务正在运行，或上次没有正常退出。")
        print()
        print("       处理办法：")
        print(f"         1) 先打开 http://127.0.0.1:{PORT} 看能不能用")
        print("         2) 不能用就跑一次「停止服务.py」，再重新启动")
        print()
        return 1
    print(f"[1/3] 端口 {PORT} 空闲，可以启动")

    # ── 2. 中间件检查 ──
    if middleware_running():
        print("[2/3] 中间件已在运行")
    else:
        print("[2/3] 中间件未启动，正在启动 Docker 容器...")
        if not start_middleware():
            return 1
        print(f"      等待 {WAIT_AFTER_MIDDLEWARE} 秒让中间件就绪...")
        import time
        time.sleep(WAIT_AFTER_MIDDLEWARE)

    # ── 3. 启动 Web 服务 ──
    print("[3/3] 正在启动 Web 服务...")
    print()
    line("-")
    print(f"  浏览器地址 :  http://127.0.0.1:{PORT}")
    print(f"  停止服务   :  按 Ctrl+C，或运行「停止服务.py」")
    line("-")
    print()
    print("  正在加载模型，约 15-20 秒，请稍候...")
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR)

    try:
        # 前台运行：这样 Ctrl+C 能直接传给它
        subprocess.run(
            [str(VENV_PYTHON), "-m", "uvicorn", "backend.main:app",
             "--host", HOST, "--port", str(PORT)],
            cwd=PROJECT_DIR, env=env)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")

    print()
    line()
    print("  服务已停止。")
    line()
    return 0


if __name__ == "__main__":
    sys.exit(main())
