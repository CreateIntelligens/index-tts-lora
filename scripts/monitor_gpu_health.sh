#!/bin/bash
# GPU 健康監控腳本 - 定期檢查容器內的 GPU，失敗就重啟

CONTAINER_NAME="index-tts-lora"
CHECK_INTERVAL=60  # 每 60 秒檢查一次

while true; do
    # 檢查容器是否在運行
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[$(date)] 容器 ${CONTAINER_NAME} 未運行"
        sleep $CHECK_INTERVAL
        continue
    fi

    # 在容器內執行 nvidia-smi 檢查
    if ! docker exec ${CONTAINER_NAME} nvidia-smi &>/dev/null; then
        echo "[$(date)] ⚠️ GPU 檢查失敗，重啟容器..."
        docker restart ${CONTAINER_NAME}
        echo "[$(date)] ✅ 容器已重啟"
    else
        echo "[$(date)] ✓ GPU 正常"
    fi

    sleep $CHECK_INTERVAL
done
