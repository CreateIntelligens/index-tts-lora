#!/bin/bash

set -e

echo "🔧 檢查並修復目錄權限..."

# 切換到工作目錄
cd /workspace/index-tts-lora

# 取得宿主機用戶 ID (從掛載的文件檢測,優先順序由高到低)
HOST_UID=""
HOST_GID=""

# 嘗試從掛載的文件取得真實的宿主機 UID
for file in docker-compose.yml Dockerfile download_models.sh webui.py train.py; do
    if [ -f "$file" ]; then
        HOST_UID=$(stat -c '%u' "$file" 2>/dev/null)
        HOST_GID=$(stat -c '%g' "$file" 2>/dev/null)
        # 如果不是 root (UID != 0), 就使用這個
        if [ "$HOST_UID" != "0" ]; then
            break
        fi
    fi
done

# 如果都找不到或都是 root, 使用預設值 1000
if [ -z "$HOST_UID" ] || [ "$HOST_UID" = "0" ]; then
    HOST_UID="1000"
    HOST_GID="1000"
fi

echo "   宿主機用戶: UID=$HOST_UID GID=$HOST_GID"

# 需要修復權限的目錄列表
DIRS_TO_FIX=(
    "checkpoints"
    "finetune_data"
    "finetune_models"
    "outputs"
    "prompts"
    "tests"
    "assets"
)

# 修復所有目錄權限
for dir in "${DIRS_TO_FIX[@]}"; do
    if [ -e "$dir" ]; then
        current_owner=$(stat -c '%u' "$dir")
        if [ "$current_owner" != "$HOST_UID" ]; then
            echo "   修復 $dir (從 UID:$current_owner 改為 $HOST_UID)"
            chown -R $HOST_UID:$HOST_GID "$dir" 2>/dev/null || true
        else
            echo "   ✓ $dir 權限正確"
        fi
    else
        echo "   - $dir 不存在,跳過"
    fi
done

echo "✅ 權限檢查完成!"
echo ""

CHECKPOINT_DIR="checkpoints"
REQUIRED_FILES=(
    "bigvgan_generator.pth"
    "bpe.model"
    "gpt.pth"
    "config.yaml"
)

echo "🔍 檢查模型文件..."

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

if [ "$all_exists" = false ]; then
    echo ""
    echo "📥 開始自動下載模型..."
    echo ""

    # 安裝 huggingface-hub (如果還沒安裝)
    pip install -q huggingface-hub

    # 下載模型
    echo "⬇️  從 HuggingFace 下載 IndexTTS-1.5 模型..."
    hf download IndexTeam/IndexTTS-1.5 --local-dir "$CHECKPOINT_DIR"

    echo ""
    echo "✅ 模型下載完成!"
else
    echo "✨ 所有模型文件已存在!"
fi

# 建立 gpt.pth.open_source 軟連結 (如果不存在)
if [ -f "$CHECKPOINT_DIR/gpt.pth" ] && [ ! -e "$CHECKPOINT_DIR/gpt.pth.open_source" ]; then
    echo "🔗 建立 gpt.pth.open_source 軟連結..."
    ln -s gpt.pth "$CHECKPOINT_DIR/gpt.pth.open_source"
    echo "✅ 軟連結建立完成"
fi

echo ""
echo "📦 安裝 indextts 套件..."

# 等待 GPU 初始化
echo "⏳ 等待 GPU 初始化..."
for i in {1..10}; do
    if nvidia-smi >/dev/null 2>&1; then
        echo "✅ GPU 已就緒"
        break
    fi
    echo "   等待中... ($i/10)"
    sleep 2
done

# 驗證 PyTorch 能看到 GPU
if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "✅ PyTorch CUDA 已就緒"
else
    echo "⚠️  警告: PyTorch 無法檢測到 CUDA，編譯可能失敗"
fi

# 自動檢測 GPU 架構並設定 TORCH_CUDA_ARCH_LIST
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ -n "$GPU_ARCH" ]; then
    echo "🎯 檢測到 GPU 架構: $GPU_ARCH"
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
else
    echo "⚠️  無法自動檢測 GPU 架構，使用預設值"
    export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
fi

# 安裝當前專案 (editable mode)
pip install -e . --no-build-isolation -q

echo "✅ indextts 套件安裝完成!"
echo ""
echo "🚀 啟動服務..."
echo ""

# 執行傳入的命令
exec "$@"
