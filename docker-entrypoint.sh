#!/bin/bash
set -e

# 動態建立與 host 相同 UID/GID 的用戶
USER_ID=${UID:-1000}
GROUP_ID=${GID:-1000}
USER_NAME=${USERNAME:-user}

# 檢查用戶是否已存在
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
    # 建立群組（如果不存在）
    if ! getent group "$GROUP_ID" >/dev/null 2>&1; then
        groupadd -g "$GROUP_ID" "$USER_NAME"
    fi

    # 建立用戶並設定 home 目錄
    useradd -m -u "$USER_ID" -g "$GROUP_ID" -s /bin/bash -d "/home/$USER_NAME" "$USER_NAME"

    # 給予 sudo 權限
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# 確保用戶 home 目錄權限正確
USER_HOME=$(getent passwd "$USER_ID" | cut -d: -f6)
if [ -d "$USER_HOME" ]; then
    chown -R "$USER_ID:$GROUP_ID" "$USER_HOME"
fi

# 建立並設定 .cache 和 .local 目錄權限
mkdir -p "$USER_HOME/.cache" "$USER_HOME/.local"
chown -R "$USER_ID:$GROUP_ID" "$USER_HOME/.cache" "$USER_HOME/.local"

# 執行傳入的命令
exec "$@"
