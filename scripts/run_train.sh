#!/bin/bash

# 訓練模型腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_common.sh"

train_model() {
    print_header "開始訓練模型"

    local use_ddp=0
    local num_gpus=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --ddp)
                use_ddp=1
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

    if [ "$use_ddp" -eq 1 ]; then
        if [ -z "$num_gpus" ]; then
            num_gpus=$(read_config "gpus" "0,1,2,3" | tr ',' ' ' | wc -w)
        fi

        print_info "使用 DDP 訓練，GPU 數量: $num_gpus"

        if [ "$USE_DOCKER" -eq 1 ]; then
            docker compose exec index-tts-lora \
                python3 -m torch.distributed.run \
                --nproc_per_node="$num_gpus" \
                tools/train_lora_ddp.py
        else
            python3 -m torch.distributed.run \
                --nproc_per_node="$num_gpus" \
                tools/train_lora_ddp.py
        fi
    else
        print_info "使用 DataParallel 訓練"

        if [ "$USE_DOCKER" -eq 1 ]; then
            docker compose exec index-tts-lora python3 tools/train_lora.py
        else
            python3 tools/train_lora.py
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
