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

    # 自動檢測 GPU 數量
    if [ -z "$num_gpus" ]; then
        if [ "$USE_DOCKER" -eq 1 ]; then
            num_gpus=$(docker compose exec index-tts-lora nvidia-smi --list-gpus 2>/dev/null | wc -l)
        else
            num_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        fi

        if [ "$num_gpus" -eq 0 ]; then
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
                train.py
        else
            python3 -m torch.distributed.run \
                --nproc_per_node="$num_gpus" \
                train.py
        fi
    else
        print_info "使用 DataParallel 訓練"

        if [ "$USE_DOCKER" -eq 1 ]; then
            docker compose exec index-tts-lora python3 train.py
        else
            python3 train.py
        fi
    fi

    if [ $? -eq 0 ]; then
        print_success "訓練完成！"
    else
        print_error "訓練失敗！"
        exit 1
    fi
}

# 直接執行時調用
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    train_model "$@"
fi
