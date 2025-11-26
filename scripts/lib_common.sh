#!/bin/bash

# 共用函數庫

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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

# 環境檢測
USE_DOCKER=1
IN_CONTAINER=0

if [ -f "/.dockerenv" ] || grep -qa 'docker' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=1
fi

if ! command -v docker >/dev/null 2>&1; then
    USE_DOCKER=0
fi

if [ "$IN_CONTAINER" -eq 1 ]; then
    USE_DOCKER=0
fi

# 配置文件路徑（統一使用 finetune_models/config.yaml）
CONFIG_FILE="finetune_models/config.yaml"

# 讀取配置（支援任意深度嵌套路徑，如 workflow.paths.data_source_dir）
read_config() {
    local key=$1
    local default=$2

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$default"
        return
    fi

    # 使用 Python 解析 YAML（支援任意深度嵌套）
    local value=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 支援點號分隔的路徑（如 workflow.paths.data_source_dir）
    keys = '$key'.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            value = None
            break
    print(value if value is not None else '$default')
except:
    print('$default')
" 2>/dev/null)

    echo "$value"
}

# 檢查容器狀態
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

export -f print_header
export -f print_info
export -f print_success
export -f print_error
export -f print_warning
export -f read_config
export -f check_container
