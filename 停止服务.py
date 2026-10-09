# -*- coding: utf-8 -*-
"""
DeliRAG 停止脚本

用法：
  双击本文件，或在终端/VSCode 里运行
      python 停止服务.py

作用：
  按端口查出 Web 服务的进程，然后结束它。

注意：本脚本只停【Web 服务】（端口 8082），不停中间件。
      中间件（Milvus/MySQL/Redis）是 Docker 容器，会继续运行、数据保留。
      要停中间件可执行：docker stop edurag-milvus edurag-mysql edurag-redis
"""

from __future__ import annotations

import subprocess
import sys

PORT = 8082


def line(char="=", n=58):
    print(char * n)


def find_pids_by_port(port: int) -> list[str]:
    """用 netstat 找出监听指定端口的所有 PID

    注意：中文 Windows 上 netstat 输出是 GBK 编码。
          如果直接 subprocess.run(text=True)，Python 会按 UTF-8 解码并抛
          UnicodeDecodeError（进而 stdout 变成 None）。
          所以这里先拿 bytes，再按可能的编码依次尝试解码。
    """
    try:
        raw = subprocess.run(["netstat", "-ano"], capture_output=True, timeout=20).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[错误] 执行 netstat 失败: {e}")
        return []

    out = None
    for enc in ("gbk", "utf-8", "cp1252"):
        try:
            out = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if out is None:
        out = raw.decode("utf-8", errors="replace")

    pids = []
    for row in out.splitlines():
        parts = row.split()
        # 形如： TCP  127.0.0.1:8081  0.0.0.0:0  LISTENING  27624
        if len(parts) >= 5 and parts[0].upper() == "TCP" \
                and f":{port}" in parts[1] and parts[3].upper() == "LISTENING":
            if parts[4] not in pids:
                pids.append(parts[4])
    return pids


def main() -> int:
    line()
    print(f"   停止 DeliRAG Web 服务（端口 {PORT}）")
    line()
    print()

    print(f"正在查找占用端口 {PORT} 的进程...")
    pids = find_pids_by_port(PORT)

    if not pids:
        print()
        print("[结果] 没有进程占用该端口，服务已经停止了。")
        print()
        print("提示：本脚本只停 Web 服务，不停中间件。")
        print("      要停中间件请执行：docker compose stop")
        print()
        return 0

    print(f"找到进程 PID = {', '.join(pids)}")
    print("正在结束...")

    failed = []
    for pid in pids:
        # 同样不传 text=True：中文 Windows 下 taskkill 的输出是 GBK
        r = subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
        if r.returncode == 0:
            print(f"  [OK] 已结束 PID {pid}")
        else:
            msg = (r.stderr or r.stdout or b"").decode("gbk", errors="replace").strip()
            print(f"  [失败] PID {pid}: {msg[:80]}")
            failed.append(pid)

    print()
    if failed:
        print("[部分失败] 可能需要用管理员身份运行本脚本。")
        return 1

    line()
    print("[成功] Web 服务已停止。")
    print("       中间件（Milvus/MySQL/Redis）仍在运行，数据保留。")
    line()
    return 0


if __name__ == "__main__":
    sys.exit(main())
