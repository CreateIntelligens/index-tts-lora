#!/bin/bash

# 準備音頻列表腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_common.sh"

prepare_audio_list() {
    if [ $# -eq 0 ]; then
        set -- "data/"
    fi

    print_header "準備音頻列表"

    # 從配置讀取 split_size
    local split_size=$(read_config "split_size" "100000")
    print_info "使用配置: split_size=${split_size}"

    # 固定參數：自動掃描 + 自動分割
    local prepare_args=("$@" "--auto-scan" "--merge-all" "--split-size" "$split_size")

    if [ "$USE_DOCKER" -eq 1 ]; then
        check_container
        if docker compose exec index-tts-lora python3 scripts/prepare_audio_list.py "${prepare_args[@]}"; then
            print_success "音頻列表準備完成！"
        else
            print_error "音頻列表準備失敗！"
            exit 1
        fi
    else
        if python3 scripts/prepare_audio_list.py "${prepare_args[@]}"; then
            print_success "音頻列表準備完成！"
        else
            print_error "音頻列表準備失敗！"
            exit 1
        fi
    fi
}

# 直接執行時調用
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    prepare_audio_list "$@"
fi
