#!/bin/bash

# IndexTTS LoRA 主控台腳本
# 所有功能已模組化到 scripts/ 目錄

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 載入共用函數
source "$SCRIPT_DIR/scripts/lib_common.sh"

# 顯示使用說明
show_help() {
    echo -e "\n${BOLD}IndexTTS LoRA 訓練主控台${NC}\n"
    echo -e "${BOLD}使用方式:${NC}"
    echo -e "  ./run.sh <命令> [選項]\n"
    echo -e "${BOLD}可用命令:${NC}"
    echo -e "  ${GREEN}prepare${NC} <data_dir>      準備音頻列表並自動分割"
    echo -e "  ${GREEN}extract${NC}                 提取音頻特徵（自動多 GPU 並行）"
    echo -e "  ${GREEN}train${NC} [--ddp] [--gpus N] [--resume PATH | --auto-resume] 訓練模型"
    echo -e "  ${GREEN}webui${NC}                   啟動 WebUI"
    echo -e "  ${GREEN}api${NC}                     啟動 FastAPI 服務 (api.py)"
    echo -e "  ${GREEN}tensorboard${NC} [logdir]     啟動 TensorBoard（背景執行，預設讀最新 logs/train_*）"
    echo -e "  ${GREEN}shell${NC}                   進入容器 shell\n"
    echo -e "${BOLD}配置文件:${NC} scripts/config.yaml / finetune_models/config.yaml\n"
    echo -e "${BOLD}範例:${NC}"
    echo -e "  ./run.sh prepare data/          # 自動掃描並按 split_size 分割"
    echo -e "  ./run.sh extract                # 自動使用所有 part 文件，多 GPU 並行"
    echo -e "  ./run.sh train --ddp --gpus 4   # 使用 4 GPU DDP 訓練"
    echo -e "  ./run.sh train --auto-resume    # 自動接續上次訓練"
    echo -e "  ./run.sh tensorboard logs/train_20251210  # 指定特定 logdir\n"
}

# 進入容器 shell
enter_shell() {
    if [ "$USE_DOCKER" -eq 1 ]; then
        print_info "進入容器 shell..."
        check_container
        docker compose exec index-tts-lora bash
    else
        if [ "$IN_CONTAINER" -eq 1 ]; then
            print_info "已在容器內"
        else
            print_info "啟動 shell..."
        fi
        bash
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
        prepare)
            source "$SCRIPT_DIR/scripts/run_prepare.sh"
            prepare_audio_list "$@"
            ;;
        extract)
            source "$SCRIPT_DIR/scripts/run_extract.sh"
            extract_features "$@"
            ;;
        train)
            source "$SCRIPT_DIR/scripts/run_train.sh"
            train_model "$@"
            ;;
        webui)
            source "$SCRIPT_DIR/scripts/run_webui.sh"
            start_webui "$@"
            ;;
        api)
            # 直接啟動 api.py（會使用 docker-compose 內映射的 7859 埠）
            if [ "$USE_DOCKER" -eq 1 ]; then
                check_container
                docker compose exec -T index-tts-lora python3 api.py --host 0.0.0.0 --port 7859
            else
                python3 api.py --host 0.0.0.0 --port 7859
            fi
            ;;
        tensorboard)
            # 先停止舊的 tensorboard
            pkill -f "tensorboard.*--port 8006" 2>/dev/null || true

            # 預設讀最新的 logs/train_*；可傳自訂 logdir
            if [ -n "$1" ]; then
                LOGDIR="$1"
            else
                latest_dir=$(ls -td logs/train_* 2>/dev/null | head -n 1)
                LOGDIR=${latest_dir:-logs}
            fi

            echo -e "${GREEN}>> 啟動 TensorBoard (背景執行)${NC}"
            echo -e "   LogDir: $LOGDIR"
            echo -e "   URL: http://0.0.0.0:8006/"

            if [ "$USE_DOCKER" -eq 1 ]; then
                check_container
                docker compose exec -d index-tts-lora bash -c "tensorboard --logdir $LOGDIR --host 0.0.0.0 --port 8006 2>/dev/null"
            else
                nohup tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port 8006 > /dev/null 2>&1 &
            fi
            echo -e "${GREEN}>> TensorBoard 已在背景啟動${NC}"
            ;;
        shell)
            enter_shell
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
