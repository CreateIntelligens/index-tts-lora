#!/bin/bash

# 訓練模型腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_common.sh"

train_model() {
    print_header "開始訓練模型"

    local mode="auto"  # auto, ddp, dp
    local num_gpus=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --ddp)
                mode="ddp"
                ;;
            --dp)
                mode="dp"
                ;;
            --gpus)
                shift
                num_gpus="$1"
                ;;
            *)
                print_warning "未知參數: $1"
                ;;
        esac
        shift
    done

    check_container

    # 建立 log 目錄
    LOG_DIR="logs/train_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/train.log"
    print_info "訓練 log 將儲存至: $LOG_FILE"

    # 自動檢測 GPU 數量（使用 PyTorch，尊重 CUDA_VISIBLE_DEVICES）
    if [ -z "$num_gpus" ]; then
        if [ "$USE_DOCKER" -eq 1 ]; then
            num_gpus=$(docker compose exec index-tts-lora python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        else
            num_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        fi

        if [ -z "$num_gpus" ] || [ "$num_gpus" -eq 0 ]; then
            print_error "找不到可用的 GPU"
            exit 1
        fi
    fi

    # 自動選擇訓練模式
    if [ "$mode" = "auto" ]; then
        if [ "$num_gpus" -gt 1 ]; then
            mode="ddp"
            print_info "🚀 檢測到 $num_gpus 個 GPU，自動使用 DDP 訓練"
        else
            mode="dp"
            print_info "📌 檢測到 $num_gpus 個 GPU，使用 DataParallel 訓練"
        fi
    fi

    if [ "$mode" = "ddp" ]; then
        print_info "使用 DDP 訓練，GPU 數量: $num_gpus"

        if [ "$USE_DOCKER" -eq 1 ]; then
            docker compose exec index-tts-lora \
                python3 -m torch.distributed.run \
                --nproc_per_node="$num_gpus" \
                train_ddp.py 2>&1 | tee "$LOG_FILE"
        else
            python3 -m torch.distributed.run \
                --nproc_per_node="$num_gpus" \
                train_ddp.py 2>&1 | tee "$LOG_FILE"
        fi
    else
        print_info "使用 DataParallel 訓練"

        if [ "$USE_DOCKER" -eq 1 ]; then
            docker compose exec index-tts-lora python3 train.py 2>&1 | tee "$LOG_FILE"
        else
            python3 train.py 2>&1 | tee "$LOG_FILE"
        fi
    fi

    # 修復 log 檔案權限
    if [ -f "$LOG_FILE" ]; then
        # 取得宿主機用戶 UID/GID
        HOST_UID=$(stat -c '%u' docker-compose.yml 2>/dev/null || echo "1000")
        HOST_GID=$(stat -c '%g' docker-compose.yml 2>/dev/null || echo "1000")
        chown $HOST_UID:$HOST_GID "$LOG_FILE" 2>/dev/null || true
        chown $HOST_UID:$HOST_GID "$LOG_DIR" 2>/dev/null || true
    fi

    if [ $? -eq 0 ]; then
        print_success "訓練完成！"
        print_info "Log 檔案: $LOG_FILE"
    else
        print_error "訓練失敗！"
        print_info "Log 檔案: $LOG_FILE"
        exit 1
    fi
}

# 直接執行時調用
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    train_model "$@"
fi
