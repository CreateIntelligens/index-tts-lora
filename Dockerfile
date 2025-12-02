# IndexTTS LoRA Docker Image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 設置環境變數
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    wget \
    build-essential \
    libsndfile1 \
    dialog \
    sudo \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# 設置 Python 3.10 為默認 python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 複製並設定 entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# 升級 pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 設置工作目錄
WORKDIR /workspace/index-tts-lora

# 複製並安裝依賴
COPY requirements.txt .

# 首先安裝 PyTorch (CUDA 12.1 版本)，然後安裝其他依賴（避免重裝 torch）
RUN pip3 install --no-cache-dir --extra-index-url ${PIP_EXTRA_INDEX_URL} \
    torch==2.1.2 torchaudio==2.1.2 && \
    pip3 install --no-cache-dir --extra-index-url ${PIP_EXTRA_INDEX_URL} \
    --no-deps -r requirements.txt && \
    pip3 install --no-cache-dir --extra-index-url ${PIP_EXTRA_INDEX_URL} \
    -r requirements.txt && \
    rm -rf /root/.cache/pip /root/.cache/torch_extensions /tmp/pip-*

# 不在映像中創建目錄,讓掛載時自動創建 (使用宿主機用戶權限)

# 複製啟動/健康檢查腳本
COPY scripts/entrypoint.sh scripts/gpu-healthcheck.sh /
RUN chmod +x /entrypoint.sh /gpu-healthcheck.sh

# 暴露 Gradio 端口
EXPOSE 7860

# 使用啟動腳本
ENTRYPOINT ["/entrypoint.sh"]

# 預設命令 (會被 docker-compose.yml 覆蓋)
CMD ["python3", "webui.py", "--host", "0.0.0.0", "--port", "7860"]
