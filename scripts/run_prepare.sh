#!/bin/bash

# 準備音頻列表腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_common.sh"

prepare_audio_list() {
    # 從配置讀取路徑（使用統一配置 finetune_models/config.yaml）
    local data_source_dir=$(read_config "workflow.paths.data_source_dir" "data")
    local audio_list_dir=$(read_config "workflow.paths.audio_list_dir" "finetune_data/audio_list")
    local split_size=$(read_config "workflow.prepare.split_size" "100000")

    # 如果有命令列參數，優先使用
    if [ $# -gt 0 ]; then
        data_source_dir="$1"
    fi

    print_header "準備音頻列表"
    print_info "資料來源: $data_source_dir"
    print_info "輸出目錄: $audio_list_dir"
    print_info "分割大小: $split_size"

    # 固定參數：自動掃描 + 合併所有 speaker + 自動分割
    local prepare_args=("$data_source_dir" "--auto-scan" "--merge-all" "--no-individual" "--split-size" "$split_size" "--output-dir" "$audio_list_dir")

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
