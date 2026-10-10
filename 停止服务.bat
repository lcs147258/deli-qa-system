@echo off
chcp 65001 >nul
title 停止 DeliRAG 服务

echo ==========================================================
echo    停止 DeliRAG Web 服务（端口 8082）
echo ==========================================================
echo.
echo 正在查找占用 8082 端口的进程...

set PID=
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort 8082 -State Listen -ErrorAction SilentlyContinue).OwningProcess"`) do set PID=%%a

if "%PID%"=="" (
    echo.
    echo [提示] 没有发现占用 8082 端口的进程，服务可能已经停止。
    echo.
    echo 提示：本脚本只停 Web 服务，不停中间件。
    echo       中间件（ParadeDB/MySQL/Redis）仍在容器中运行、数据保留。
    echo       要停中间件：docker stop edurag-paradedb edurag-mysql edurag-redis
    echo.
    pause
    exit /b 0
)

echo 找到服务进程 PID = %PID%
echo 正在结束进程...
taskkill /PID %PID% /F

if errorlevel 1 (
    echo.
    echo [失败] 无法结束该进程，请用管理员身份运行本脚本
) else (
    echo.
    echo [成功] Web 服务已停止。
    echo        中间件仍在运行（ParadeDB/MySQL/Redis），数据不受影响。
)
echo.
pause
