#!/bin/bash

# 提取音頻特徵腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_common.sh"

# 單個 GPU 執行 extract
run_single_extraction() {
    local audio_list="$1"
    local gpu_id="$2"
    local batch_size="$3"
    local num_workers="$4"
    local log_file="$5"

    if [ "$USE_DOCKER" -eq 1 ]; then
        docker compose exec -T \
            -e CUDA_VISIBLE_DEVICES="$gpu_id" \
            index-tts-lora \
            python3 tools/extract_codec.py \
            --audio_list "$audio_list" \
            --extract_condition \
            --device cuda \
            --batch_size "$batch_size" \
            --num_workers "$num_workers" \
            > "$log_file" 2>&1
    else
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        python3 tools/extract_codec.py \
            --audio_list "$audio_list" \
            --extract_condition \
            --device cuda \
            --batch_size "$batch_size" \
            --num_workers "$num_workers" \
            > "$log_file" 2>&1
    fi
}

# GPU worker 進程
run_gpu_worker() {
    local gpu_id=$1
    local task_file=$2
    local task_lock=$3
    local stats_file=$4
    local batch_size=$5
    local num_workers=$6
    local log_dir=$7

    while true; do
        local audio_list=""
        local tmp_task=$(mktemp)

        # 使用 flock 確保原子操作
        (
            flock -x 200

            if [ -s "$task_file" ]; then
                head -n 1 "$task_file" > "$tmp_task"
                sed -i '1d' "$task_file"
            fi
        ) 200>"$task_lock"

        # 從臨時文件讀取任務
        if [ -s "$tmp_task" ]; then
            audio_list=$(cat "$tmp_task")
        fi
        rm -f "$tmp_task"

        if [ -z "$audio_list" ]; then
            break
        fi

        local log_file="${log_dir}/gpu${gpu_id}_$(basename "$audio_list" .txt).log"
        print_info "[GPU $gpu_id] 開始處理: $(basename "$audio_list") (日誌: $log_file)"

        if run_single_extraction "$audio_list" "$gpu_id" "$batch_size" "$num_workers" "$log_file"; then
            (
                flock -x 200
                echo "[GPU $gpu_id] ✓ $(basename "$audio_list")" >> "$stats_file"
            ) 200>"$task_lock"
            print_success "[GPU $gpu_id] 完成: $(basename "$audio_list")"
        else
            (
                flock -x 200
                echo "[GPU $gpu_id] ✗ $(basename "$audio_list")" >> "$stats_file"
            ) 200>"$task_lock"
            print_error "[GPU $gpu_id] 失敗: $(basename "$audio_list") (查看日誌: $log_file)"
        fi
    done
}

# 提取特徵
extract_features() {
    print_header "提取音頻特徵"

    local base_dir="finetune_data/audio_list"

    # 從配置讀取所有參數
    local batch_size=$(read_config "batch_size" "32")
    local num_workers=$(read_config "num_workers" "12")
    local gpu_csv=$(read_config "gpus" "")

    print_info "使用配置: batch_size=${batch_size}, num_workers=${num_workers}"

    # 檢查目錄
    if [ ! -d "$base_dir" ]; then
        print_error "找不到 $base_dir 目錄，請先執行 ./run.sh prepare"
        exit 1
    fi

    # 自動查找所有 part 文件
    local audio_lists=()
    shopt -s nullglob
    for file in "$base_dir"/audio_list_part_*.txt; do
        audio_lists+=("$file")
    done

    # 如果沒有 part 文件，查找其他 txt 文件
    if [ "${#audio_lists[@]}" -eq 0 ]; then
        for file in "$base_dir"/*.txt; do
            audio_lists+=("$file")
        done
    fi
    shopt -u nullglob

    if [ "${#audio_lists[@]}" -eq 0 ]; then
        print_error "找不到任何 audio_list 檔案"
        exit 1
    fi

    # 解析 GPU 配置
    local -a gpu_ids=()
    if [ -n "$gpu_csv" ]; then
        IFS=',' read -r -a gpu_ids <<< "${gpu_csv// /}"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES// /}"
    else
        # 預設使用 GPU 0
        gpu_ids=(0)
    fi

    if [ "${#gpu_ids[@]}" -eq 0 ]; then
        gpu_ids=(0)
    fi

    print_info "發現 ${#audio_lists[@]} 個檔案，使用 ${#gpu_ids[@]} 張 GPU (${gpu_ids[*]})"

    # 創建日誌目錄
    local log_dir="logs/extract_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$log_dir"
    print_info "日誌目錄: $log_dir"

    # 創建任務文件
    local task_file=$(mktemp)
    local task_lock="${task_file}.lock"
    local stats_file=$(mktemp)

    for audio_list in "${audio_lists[@]}"; do
        echo "$audio_list" >> "$task_file"
    done

    # 啟動 GPU workers
    local -a pids=()
    for gpu_id in "${gpu_ids[@]}"; do
        run_gpu_worker "$gpu_id" "$task_file" "$task_lock" "$stats_file" "$batch_size" "$num_workers" "$log_dir" &
        pids+=($!)
    done

    echo ""
    print_info "監控進度: tail -f $log_dir/*.log"
    echo ""

    # 等待所有 worker 完成
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    # 顯示統計
    echo ""
    print_header "提取結果統計"
    if [ -f "$stats_file" ]; then
        cat "$stats_file"
        local success_count=$(grep -c "✓" "$stats_file" || true)
        local fail_count=$(grep -c "✗" "$stats_file" || true)
        echo ""
        print_success "成功: $success_count 個"
        if [ "$fail_count" -gt 0 ]; then
            print_error "失敗: $fail_count 個"
            failed=1
        fi
    fi

    # 清理臨時文件
    rm -f "$task_file" "$task_lock" "$stats_file"

    if [ "$failed" -eq 1 ]; then
        print_error "部分提取失敗"
        exit 1
    fi

    print_success "特徵提取完成！"
}

# 直接執行時調用
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    extract_features "$@"
fi
