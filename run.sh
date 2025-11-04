#!/bin/bash

# IndexTTS LoRA 主控台腳本
# 統一管理所有訓練流程

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

USE_DOCKER=1
IN_CONTAINER=0

if [ -f "/.dockerenv" ] || grep -qa 'docker' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=1
fi

# 列印函數
print_header() {
    local separator=$(printf '=%.0s' {1..60})
    echo -e "\n${BOLD}${BLUE}${separator}${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}${separator}${NC}\n"
}

print_info() {
    echo -e "${CYAN}ℹ ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

print_error() {
    echo -e "${RED}✗ ${1}${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

if ! command -v docker >/dev/null 2>&1; then
    USE_DOCKER=0
fi

# 顯示使用說明
show_help() {
    echo -e "\n${BOLD}IndexTTS LoRA 訓練主控台${NC}\n"
    echo -e "${BOLD}使用方式:${NC}"
    echo -e "  ./run.sh <命令> [選項]\n"
    echo -e "${BOLD}可用命令:${NC}"
    echo -e "  ${GREEN}pipeline${NC} <data_dir>     執行完整訓練流程"
    echo -e "  ${GREEN}prepare${NC} <data_dir>      準備音頻列表"
    echo -e "  ${GREEN}extract${NC}                 提取音頻特徵"
    echo -e "  ${GREEN}train${NC}                   訓練模型"
    echo -e "  ${GREEN}webui${NC}                   啟動 WebUI"
    echo -e "  ${GREEN}shell${NC}                   進入容器 shell"
    echo -e "  ${GREEN}logs${NC}                    查看容器日誌"
    echo -e "  ${GREEN}status${NC}                  查看容器狀態\n"
    echo -e "${BOLD}範例:${NC}"
    echo -e "  # 準備音頻列表"
    echo -e "  ./run.sh prepare data/                 # 自動掃描所有說話人"
    echo -e "  ./run.sh prepare data/080 data/081     # 多說話人\n"
    echo -e "  # 提取特徵和訓練"
    echo -e "  ./run.sh extract                       # 提取特徵 (自動判斷列表)"
    echo -e "  ./run.sh extract finetune_data/080.txt # 指定說話人列表提取特徵"
    echo -e "  ./run.sh extract --merge-all           # 合併所有列表後提取特徵"
    echo -e "  ./run.sh train                         # 訓練模型\n"
    echo -e "  # 其他命令"
    echo -e "  ./run.sh webui                         # 啟動 WebUI"
    echo -e "  ./run.sh shell                         # 進入容器"
    echo -e "  ./run.sh logs                          # 查看日誌\n"
}

# 檢查 Docker 容器狀態
check_container() {
    if [ "$USE_DOCKER" -eq 0 ]; then
        return
    fi

    if ! docker compose ps | grep -q "index-tts-lora.*Up"; then
        print_warning "容器未運行，正在啟動..."
        docker compose up -d
        sleep 5
    fi
}

ensure_audio_list_file() {
    local override="$1"
    local merge_all="$2"
    local base_dir="finetune_data/audio_list"
    local default_file="$base_dir/audio_list.txt"

    if [ -n "$override" ]; then
        if [ -f "$override" ]; then
            echo "$override"
            return 0
        elif [ -f "$base_dir/$override" ]; then
            echo "$base_dir/$override"
            return 0
        else
            print_error "指定的列表不存在: $override"
            exit 1
        fi
    fi

    if [ ! -d "$base_dir" ]; then
        print_error "找不到 finetune_data/audio_list 目錄，請先執行 ./run.sh prepare"
        exit 1
    fi

    local speaker_lists=()
    shopt -s nullglob
    for file in "$base_dir"/*.txt; do
        local name="${file##*/}"
        if [ "$name" = "audio_list.txt" ] || [ "$name" = "audio_list_all.txt" ]; then
            continue
        fi
        speaker_lists+=("$file")
    done
    shopt -u nullglob

    if [ "$merge_all" -eq 1 ]; then
        if [ "${#speaker_lists[@]}" -eq 0 ]; then
            print_error "沒有可合併的列表，請先執行 ./run.sh prepare"
            exit 1
        fi

        print_info "合併 ${#speaker_lists[@]} 個列表生成 $default_file..." >&2
        if ! python3 - "$default_file" "${speaker_lists[@]}" <<'PY'; then
from pathlib import Path
from scripts.prepare_audio_list import fix_file_permissions
import sys

target = Path(sys.argv[1])
sources = [Path(p) for p in sys.argv[2:]]

lines: list[str] = []
for src in sources:
    with src.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                lines.append(raw)

target.write_text("\n".join(lines), encoding="utf-8")
fix_file_permissions(target)
PY
            print_error "自動生成 audio_list.txt 失敗，請確認 finetune_data 內容"
            exit 1
        fi
        echo "$default_file"
        return 0
    fi

    if [ -f "$default_file" ] && [ "${#speaker_lists[@]}" -le 1 ]; then
        echo "$default_file"
        return 0
    fi

    if [ "${#speaker_lists[@]}" -eq 0 ]; then
        print_error "沒有可用的音頻列表，請先執行 ./run.sh prepare"
        exit 1
    fi

    if [ "${#speaker_lists[@]}" -eq 1 ]; then
        echo "${speaker_lists[0]}"
        return 0
    fi

    print_error "偵測到多個說話人列表，請指定要使用的檔案或加入 --merge-all"
    for list in "${speaker_lists[@]}"; do
        echo "   - $list" >&2
    done
    exit 1
}

# 準備音頻列表 (單說話人或多說話人)
prepare_audio_list() {
    if [ $# -eq 0 ]; then
        # 沒有參數時預設使用 data/
        set -- "data/"
    fi

    print_header "準備音頻列表"
    if [ "$USE_DOCKER" -eq 1 ]; then
        check_container
        if docker compose exec index-tts-lora python3 scripts/prepare_audio_list.py "$@"; then
            print_success "音頻列表準備完成！"
        else
            print_error "音頻列表準備失敗！"
            exit 1
        fi
    else
        if [ "$IN_CONTAINER" -eq 1 ]; then
            print_info "容器內環境下直接執行音頻列表準備..."
        else
            print_warning "未檢測到 Docker，直接在當前環境執行音頻列表準備..."
        fi
        if python3 scripts/prepare_audio_list.py "$@"; then
            print_success "音頻列表準備完成！"
        else
            print_error "音頻列表準備失敗！"
            exit 1
        fi
    fi
}

# 提取特徵
extract_features() {
    print_header "提取音頻特徵"

    local base_dir="finetune_data/audio_list"

    if [ ! -d "$base_dir" ]; then
        print_error "找不到 $base_dir 目錄，請先執行 ./run.sh prepare"
        exit 1
    fi

    # 掃描所有 audio_list txt 檔案
    local audio_lists=()
    shopt -s nullglob
    for file in "$base_dir"/*.txt; do
        audio_lists+=("$file")
    done
    shopt -u nullglob

    if [ "${#audio_lists[@]}" -eq 0 ]; then
        print_error "在 $base_dir 找不到任何 audio_list 檔案"
        exit 1
    fi

    print_info "發現 ${#audio_lists[@]} 個 audio_list 檔案"

    local success_count=0
    local fail_count=0

    for audio_list in "${audio_lists[@]}"; do
        local filename=$(basename "$audio_list")
        echo ""
        print_info "處理: $filename"

        if [ "$USE_DOCKER" -eq 1 ]; then
            check_container
            if docker compose exec index-tts-lora python3 tools/extract_codec.py \
                --audio_list "$audio_list" \
                --extract_condition; then
                print_success "✓ $filename 處理完成"
                success_count=$((success_count + 1))
            else
                print_error "✗ $filename 處理失敗"
                fail_count=$((fail_count + 1))
            fi
        else
            if [ "$IN_CONTAINER" -eq 1 ]; then
                print_info "容器內環境下直接執行特徵提取..."
            else
                print_warning "未檢測到 Docker，直接在當前環境執行特徵提取..."
            fi
            if python3 tools/extract_codec.py \
                --audio_list "$audio_list" \
                --extract_condition; then
                print_success "✓ $filename 處理完成"
                success_count=$((success_count + 1))
            else
                print_error "✗ $filename 處理失敗"
                fail_count=$((fail_count + 1))
            fi
        fi
    done

    echo ""
    print_info "處理結果: 成功 $success_count 個，失敗 $fail_count 個"

    if [ "$fail_count" -gt 0 ]; then
        print_error "部分檔案處理失敗！"
        exit 1
    else
        print_success "所有特徵提取完成！"
    fi
}

# 訓練模型
train_model() {
    print_header "訓練模型"
    print_info "開始訓練，這可能需要較長時間..."

    if [ "$USE_DOCKER" -eq 1 ]; then
        check_container
        if docker compose exec index-tts-lora python3 train.py; then
            print_success "模型訓練完成！"
        else
            print_error "模型訓練失敗！"
            exit 1
        fi
    else
        if [ "$IN_CONTAINER" -eq 1 ]; then
            print_info "容器內環境下直接執行訓練..."
        else
            print_warning "未檢測到 Docker，直接在當前環境執行訓練..."
        fi
        if python3 train.py; then
            print_success "模型訓練完成！"
        else
            print_error "模型訓練失敗！"
            exit 1
        fi
    fi
}

# 啟動 WebUI
start_webui() {
    print_header "啟動 WebUI"
    print_info "正在啟動 WebUI..."
    print_info "訪問地址: http://localhost:7860"
    print_warning "按 Ctrl+C 停止"

    if [ "$USE_DOCKER" -eq 1 ]; then
        check_container
        docker compose exec index-tts-lora python3 webui.py \
            --host 0.0.0.0 \
            --port 7860
    else
        if [ "$IN_CONTAINER" -eq 1 ]; then
            print_info "容器內環境下直接啟動 WebUI..."
        else
            print_warning "未檢測到 Docker，直接在當前環境啟動 WebUI..."
        fi
        python3 webui.py \
            --host 0.0.0.0 \
            --port 7860
    fi
}

# 完整流程
full_pipeline() {
    local data_dir=$1

    if [ -z "$data_dir" ]; then
        print_error "請指定數據目錄！"
        echo "使用方式: ./run.sh pipeline <data_dir>"
        exit 1
    fi

    print_header "完整訓練流程"
    echo -e "${BOLD}流程步驟:${NC}"
    echo "  1. 準備音頻列表"
    echo "  2. 提取音頻特徵"
    echo "  3. 訓練模型"
    echo ""

    # 步驟 1
    prepare_audio_list "$data_dir"

    # 步驟 2
    extract_features --merge-all

    # 步驟 3
    train_model

    print_success "完整訓練流程執行成功！"
}

# 進入容器 shell
enter_shell() {
    if [ "$USE_DOCKER" -eq 1 ]; then
        print_info "進入容器 shell..."
        check_container
        docker compose exec index-tts-lora bash
    else
        if [ "$IN_CONTAINER" -eq 1 ]; then
            print_info "容器內環境，直接使用當前 shell..."
        else
            print_warning "未檢測到 Docker，直接開啟當前環境的 shell..."
        fi
        bash
    fi
}

# 查看日誌
show_logs() {
    if [ "$USE_DOCKER" -eq 1 ]; then
        print_info "顯示容器日誌..."
        docker compose logs -f index-tts-lora
    else
        print_error "當前環境無法使用 docker logs。"
        exit 1
    fi
}

# 查看狀態
show_status() {
    print_header "容器狀態"
    if [ "$USE_DOCKER" -eq 0 ]; then
        if [ "$IN_CONTAINER" -eq 1 ]; then
            print_info "容器內環境，跳過 docker compose 狀態檢查。"
        else
            print_warning "未檢測到 Docker，無法檢查 docker compose 狀態。"
        fi
        return
    fi

    docker compose ps
    echo ""

    if docker compose ps | grep -q "index-tts-lora.*Up"; then
        print_success "容器正在運行"

        # 檢查 indextts 是否已安裝
        echo ""
        print_info "檢查 indextts 套件..."
        if docker compose exec -T index-tts-lora python3 -c "import indextts" 2>/dev/null; then
            print_success "indextts 套件已安裝"
        else
            print_warning "indextts 套件未安裝或正在安裝中"
        fi

        # 檢查模型文件
        echo ""
        print_info "檢查模型文件..."
        docker compose exec -T index-tts-lora ls -l checkpoints/ | grep -E "(gpt|dvae|bigvgan)" | awk '{print "  "$9, "("$5" bytes)"}'

        # 檢查數據文件
        echo ""
        print_info "檢查數據文件..."
        if docker compose exec -T index-tts-lora [ -f finetune_data/audio_list.txt ]; then
            local count=$(docker compose exec -T index-tts-lora wc -l < finetune_data/audio_list.txt)
            print_success "audio_list.txt 存在 ($count 條音頻)"
        else
            print_warning "audio_list.txt 不存在"
        fi
    else
        print_warning "容器未運行"
    fi
}

# 主程式
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    local command=$1
    shift

    case "$command" in
        pipeline)
            full_pipeline "$@"
            ;;
        prepare)
            prepare_audio_list "$@"
            ;;
        extract)
            extract_features "$@"
            ;;
        train)
            train_model
            ;;
        webui)
            start_webui
            ;;
        shell)
            enter_shell
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
