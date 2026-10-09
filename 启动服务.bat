@echo off
chcp 65001 >nul
cd /d "%~dp0"
title DeliRAG 熟食问答服务（关闭本窗口即停止服务）

echo ==========================================================
echo    DeliRAG 熟食问答系统 - 启动
echo ==========================================================
echo.

rem ---------- 1. 检测端口是否已被占用 ----------
netstat -ano | findstr ":8082" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [警告] 端口 8082 已被占用！
    echo.
    echo         可能服务正在运行，或上次没有正常退出。
    echo.
    echo         处理办法：
    echo           1^) 先确认 http://127.0.0.1:8082 能不能正常访问
    echo           2^) 不能用就双击 "停止服务.bat" 清理后再重新启动
    echo.
    pause
    exit /b 1
)
echo [1/3] 端口 8082 空闲，可以启动

rem ---------- 2. 启动中间件（Milvus/MySQL/Redis，与教育版共用容器） ----------
docker ps --format "{{.Names}}" 2>nul | findstr "edurag-mysql" >nul 2>&1
if errorlevel 1 (
    echo [2/3] 中间件未运行，正在启动 Docker 容器...
    echo       ^(若 Docker Desktop 没启动会第一次失败^)
    docker start edurag-etcd edurag-minio edurag-milvus edurag-redis edurag-mysql
    if errorlevel 1 (
        echo       容器不存在，改用 docker compose 创建...
        docker compose up -d
    )
    echo       等待 35 秒让中间件就绪...
    timeout /t 35 /nobreak >nul
) else (
    echo [2/3] 中间件已在运行
)

rem ---------- 3. 启动 Web 服务 ----------
set PYTHONPATH=%~dp0

echo [3/3] 正在启动 Web 服务...
echo.
echo ----------------------------------------------------------
echo   浏览器地址 :  http://127.0.0.1:8082
echo   停止服务   :  按 Ctrl+C，或直接关闭本窗口
echo ----------------------------------------------------------
echo.
echo   首次加载模型约 15-20 秒，请稍候...
echo.

"E:\venvs\edurag\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8082

echo.
echo ==========================================================
echo   服务已停止
echo ==========================================================
pause
