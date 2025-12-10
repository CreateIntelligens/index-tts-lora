#!/bin/bash

# 啟動 API Server (背景執行)
echo ">> [系統] 正在啟動 API Server (Port 7859)..."
python3 api.py --host 0.0.0.0 --port 7859 &
API_PID=$!

# 等待幾秒讓 API 先初始化（可選）
sleep 5

# 啟動 WebUI (前台執行)
echo ">> [系統] 正在啟動 WebUI (Port 7860)..."
python3 webui.py --host 0.0.0.0 --port 7860 &
WEBUI_PID=$!

# 捕捉訊號以優雅關閉
trap "kill $API_PID $WEBUI_PID; exit" SIGINT SIGTERM

# 等待所有子進程
wait
