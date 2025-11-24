#!/bin/bash

# 啟動 WebUI 腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_common.sh"

start_webui() {
    print_header "啟動 WebUI"

    check_container

    if [ "$USE_DOCKER" -eq 1 ]; then
        print_info "WebUI 運行在 http://0.0.0.0:7860"
        docker compose exec index-tts-lora python3 tools/webui.py
    else
        print_info "WebUI 運行在 http://0.0.0.0:7860"
        python3 tools/webui.py
    fi
}

# 直接執行時調用
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    start_webui "$@"
fi
