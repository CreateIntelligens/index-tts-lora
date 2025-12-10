#!/bin/bash

# 訓練模型腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_common.sh"

# TensorBoard 啟動函數
start_tensorboard() {
    local port="${1:-7859}"
    local logdir="${2:-logs}"

    print_header "啟動 TensorBoard"
    print_info "Log 目錄: $logdir"
    print_info "Port: $port"

    check_container

    if [ "$USE_DOCKER" -eq 1 ]; then
        # 檢查是否已有 TensorBoard 在運行
        local existing=$(docker compose exec -T index-tts-lora pgrep -f "tensorboard" 2>/dev/null)
        if [ -n "$existing" ]; then
            print_warning "TensorBoard 已在運行中 (PID: $existing)"
            print_info "存取網址: http://localhost:$port"
            return 0
        fi

        print_info "在 Docker 容器內啟動 TensorBoard..."
        docker compose exec -d index-tts-lora tensorboard \
            --logdir="/workspace/index-tts-lora/$logdir" \
            --host=0.0.0.0 \
            --port="$port"
    else
        local existing=$(pgrep -f "tensorboard")
        if [ -n "$existing" ]; then
            print_warning "TensorBoard 已在運行中 (PID: $existing)"
            print_info "存取網址: http://localhost:$port"
            return 0
        fi

        tensorboard --logdir="$logdir" --host=0.0.0.0 --port="$port" &
    fi

    sleep 2
    print_success "TensorBoard 已啟動"
    print_info "存取網址: http://localhost:$port"
}

# 停止 TensorBoard
stop_tensorboard() {
    print_header "停止 TensorBoard"

    check_container

    if [ "$USE_DOCKER" -eq 1 ]; then
        docker compose exec -T index-tts-lora pkill -f "tensorboard" 2>/dev/null || true
    else
        pkill -f "tensorboard" 2>/dev/null || true
    fi

    print_success "TensorBoard 已停止"
}

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
    RUN_NAME="train_$(date +%Y%m%d_%H%M%S)"
    LOG_DIR="logs/${RUN_NAME}"
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

        export NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_BLOCKING_WAIT=1
        export NCCL_DEBUG=INFO
        export RUN_NAME

        if [ "$USE_DOCKER" -eq 1 ]; then
            # 使用 docker exec 並同時導向容器的 stdout（會出現在 docker logs）
            docker compose exec -T index-tts-lora bash -c "
                export RUN_NAME='$RUN_NAME'
                export RUN_LOG_DIR='/workspace/index-tts-lora/$LOG_DIR'
                python3 -m torch.distributed.run \
                --nproc_per_node=$num_gpus \
                train_ddp.py 2>&1 | tee /workspace/index-tts-lora/$LOG_FILE | tee /proc/1/fd/1
            " 2>&1 | tee "$LOG_FILE"
        else
            RUN_NAME="$RUN_NAME" RUN_LOG_DIR="$LOG_DIR" python3 -m torch.distributed.run \
                --nproc_per_node="$num_gpus" \
                train_ddp.py 2>&1 | tee "$LOG_FILE"
        fi
    else
        print_info "使用 DataParallel 訓練"

        if [ "$USE_DOCKER" -eq 1 ]; then
            docker compose exec -T index-tts-lora bash -c "
                export RUN_NAME='$RUN_NAME'
                export RUN_LOG_DIR='/workspace/index-tts-lora/$LOG_DIR'
                python3 train.py 2>&1 | tee /workspace/index-tts-lora/$LOG_FILE | tee /proc/1/fd/1
            " 2>&1 | tee "$LOG_FILE"
        else
            RUN_NAME="$RUN_NAME" RUN_LOG_DIR="$LOG_DIR" python3 train.py 2>&1 | tee "$LOG_FILE"
        fi
    fi

    # 修復 log 檔案權限
    if [ -f "$LOG_FILE" ]; then
        # 取得宿主機用戶 UID/GID
        HOST_UID=$(stat -c '%u' docker-compose.yml 2>/dev/null || echo "1000")
        HOST_GID=$(stat -c '%g' docker-compose.yml 2>/dev/null || echo "1000")
        chown -R $HOST_UID:$HOST_GID "$LOG_DIR" 2>/dev/null || true
    fi

    # 嘗試修復訓練輸出目錄權限（checkpoints 等）
    if [ -d "finetune_models" ]; then
        HOST_UID=${HOST_UID:-$(id -u)}
        HOST_GID=${HOST_GID:-$(id -g)}
        chown -R $HOST_UID:$HOST_GID finetune_models 2>/dev/null || true
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

show_usage() {
    echo "用法: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train [--ddp|--dp] [--gpus N]  開始訓練模型"
    echo "  tensorboard [--port PORT]      啟動 TensorBoard"
    echo "  tensorboard-stop               停止 TensorBoard"
    echo ""
    echo "Examples:"
    echo "  $0 train                       自動選擇訓練模式"
    echo "  $0 train --ddp --gpus 4        使用 4 個 GPU 進行 DDP 訓練"
    echo "  $0 tensorboard                 啟動 TensorBoard (預設 port 7859)"
    echo "  $0 tensorboard --port 8080     指定 port 啟動 TensorBoard"
    echo "  $0 tensorboard-stop            停止 TensorBoard"
}

# 直接執行時調用
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    case "${1:-}" in
        train)
            shift
            train_model "$@"
            ;;
        tensorboard)
            shift
            port="7859"
            logdir="logs"
            while [ $# -gt 0 ]; do
                case "$1" in
                    --port)
                        shift
                        port="$1"
                        ;;
                    --logdir)
                        shift
                        logdir="$1"
                        ;;
                    *)
                        print_warning "未知參數: $1"
                        ;;
                esac
                shift
            done
            start_tensorboard "$port" "$logdir"
            ;;
        tensorboard-stop)
            stop_tensorboard
            ;;
        -h|--help|help)
            show_usage
            ;;
        "")
            # 預設行為：開始訓練
            train_model "$@"
            ;;
        *)
            print_error "未知指令: $1"
            show_usage
            exit 1
            ;;
    esac
fi
