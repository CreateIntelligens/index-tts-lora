#!/bin/bash

set -e

CHECKPOINT_DIR="checkpoints"
REQUIRED_FILES=(
    "bigvgan_generator.pth"
    "bpe.model"
    "gpt.pth"
    "config.yaml"
)

echo "🔍 檢查模型文件..."

# 創建 checkpoints 目錄
mkdir -p "$CHECKPOINT_DIR"

# 檢查所有必需文件是否存在
all_exists=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CHECKPOINT_DIR/$file" ]; then
        echo "❌ 缺少文件: $file"
        all_exists=false
    else
        echo "✅ 已存在: $file"
    fi
done

if [ "$all_exists" = true ]; then
    echo ""
    echo "✨ 所有模型文件已存在,無需下載!"
    exit 0
fi

echo ""
echo "📥 開始下載模型..."
echo ""

# 檢查是否安裝 huggingface-hub
if ! command -v hf &> /dev/null; then
    echo "📦 安裝 huggingface-hub..."
    pip install -q huggingface-hub
fi

# 確保 checkpoints 目錄有寫入權限
chmod 755 "$CHECKPOINT_DIR" 2>/dev/null || true

# 下載模型 (使用新的 hf 命令)
echo "⬇️  從 HuggingFace 下載 IndexTTS-1.5 模型..."
echo "   來源: IndexTeam/IndexTTS-1.5"
echo ""

# 使用絕對路徑避免權限問題
CHECKPOINT_ABS=$(cd "$CHECKPOINT_DIR" && pwd)
hf download IndexTeam/IndexTTS-1.5 --local-dir "$CHECKPOINT_ABS" --repo-type model

echo ""
echo "✅ 模型下載完成!"
echo ""
echo "📂 文件位置: $CHECKPOINT_DIR/"
ls -lh "$CHECKPOINT_DIR"
