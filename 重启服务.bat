@echo off
chcp 936 >nul
title 重启 EduRAG 服务

echo ==========================================================
echo    重启 EduRAG Web 服务
echo ==========================================================
echo.
echo 第 1 步：停止现有服务
echo.
call "%~dp0停止服务.bat"
echo.
echo 第 2 步：重新启动
echo.
call "%~dp0启动服务.bat"
