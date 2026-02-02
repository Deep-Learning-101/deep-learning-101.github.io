#!/usr/bin/env bash
set -euo pipefail

# 取得腳本所在目錄
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"
EXTRA_COMPOSE_FILE="$ROOT_DIR/docker-compose.extra.yml"
IMAGE_NAME="${OPENCLAW_IMAGE:-openclaw:local}"
DOCKERIGNORE_FILE="$ROOT_DIR/.dockerignore"

echo "==> Checking configuration..."

# 1. 強制修正 .dockerignore (移除 vendor 限制)
if [ -f "$DOCKERIGNORE_FILE" ]; then
    if grep -q "^vendor" "$DOCKERIGNORE_FILE"; then
        echo "Patching .dockerignore..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' '/^vendor/d' "$DOCKERIGNORE_FILE"
        else
            sed -i '/^vendor/d' "$DOCKERIGNORE_FILE"
        fi
    fi
fi

# 2. 載入環境變數
if [ ! -f "$ENV_FILE" ]; then
    touch "$ENV_FILE"
fi

# 確保 UID/GID 寫入
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)
if ! grep -q "^HOST_UID=" "$ENV_FILE"; then echo "HOST_UID=$CURRENT_UID" >> "$ENV_FILE"; fi
if ! grep -q "^HOST_GID=" "$ENV_FILE"; then echo "HOST_GID=$CURRENT_GID" >> "$ENV_FILE"; fi

# 確保 Token 存在
if ! grep -q "^OPENCLAW_GATEWAY_TOKEN=" "$ENV_FILE"; then
    echo "Generating new Token..."
    echo "OPENCLAW_GATEWAY_TOKEN=$(openssl rand -hex 32)" >> "$ENV_FILE"
fi

# 確保目錄變數
if ! grep -q "^OPENCLAW_CONFIG_DIR=" "$ENV_FILE"; then echo "OPENCLAW_CONFIG_DIR=${HOME}/.openclaw" >> "$ENV_FILE"; fi
if ! grep -q "^OPENCLAW_WORKSPACE_DIR=" "$ENV_FILE"; then echo "OPENCLAW_WORKSPACE_DIR=${HOME}/.openclaw/workspace" >> "$ENV_FILE"; fi

# 載入變數
set -a
source "$ENV_FILE"
set +a

# 建立目錄
mkdir -p "$OPENCLAW_CONFIG_DIR"
mkdir -p "$OPENCLAW_WORKSPACE_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing dependency: $1" >&2
    exit 1
  fi
}

require_cmd docker

COMPOSE_ARGS=("-f" "$COMPOSE_FILE")
if [ -f "$EXTRA_COMPOSE_FILE" ]; then
  COMPOSE_ARGS+=("-f" "$EXTRA_COMPOSE_FILE")
fi

export OPENCLAW_DOCKER_APT_PACKAGES="${OPENCLAW_DOCKER_APT_PACKAGES:-}"

echo "==> Building Docker image..."
docker build --no-cache \
  --build-arg "OPENCLAW_DOCKER_APT_PACKAGES=${OPENCLAW_DOCKER_APT_PACKAGES}" \
  -t "$IMAGE_NAME" \
  -f "$ROOT_DIR/Dockerfile" \
  "$ROOT_DIR"

echo ""
echo "==> Onboarding..."
# 先執行標準 onboard (這會產生隨機 Token 的檔案)
docker compose "${COMPOSE_ARGS[@]}" run --rm openclaw-cli onboard --no-install-daemon

# --- [關鍵修正] 強制覆蓋 Token ---
echo "==> Enforcing Token from .env..."
CONFIG_JSON="$OPENCLAW_CONFIG_DIR/openclaw.json"

if [ -f "$CONFIG_JSON" ]; then
    echo "Patching $CONFIG_JSON with token from .env..."
    # 使用 sed 強制替換 json 檔案裡的 token 值
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/\"token\": \".*\"/\"token\": \"$OPENCLAW_GATEWAY_TOKEN\"/" "$CONFIG_JSON"
    else
        sed -i "s/\"token\": \".*\"/\"token\": \"$OPENCLAW_GATEWAY_TOKEN\"/" "$CONFIG_JSON"
    fi
    echo "Token has been forcibly updated to match .env"
else
    echo "Warning: Config file not found, skipping token patch."
fi
# --------------------------------

echo ""
echo "==> Starting gateway..."
docker compose "${COMPOSE_ARGS[@]}" up -d openclaw-gateway

echo ""
echo "=================================================="
echo "Gateway is running."
echo "Token: $OPENCLAW_GATEWAY_TOKEN"
echo "=================================================="