FROM node:22-bookworm

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

RUN corepack enable

WORKDIR /app

# 安裝系統依賴
ARG OPENCLAW_DOCKER_APT_PACKAGES=""
RUN if [ -n "$OPENCLAW_DOCKER_APT_PACKAGES" ]; then \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $OPENCLAW_DOCKER_APT_PACKAGES && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
    fi

# 安裝 gogcli
RUN mkdir -p /tmp/gog_install && \
    curl -L https://github.com/steipete/gogcli/releases/download/v0.9.0/gogcli_0.9.0_linux_amd64.tar.gz | tar -xz -C /tmp/gog_install && \
    find /tmp/gog_install -type f \( -name "gog" -o -name "gogcli" \) -exec mv {} /usr/local/bin/gog \; && \
    chmod +x /usr/local/bin/gog && \
    ln -sf /usr/local/bin/gog /usr/local/bin/gogcli && \
    rm -rf /tmp/gog_install

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY ui/package.json ./ui/package.json
COPY patches ./patches
COPY scripts ./scripts

# 第一次安裝
RUN pnpm install --frozen-lockfile

# 複製所有原始碼
COPY . .

# [除錯] 確保 vendor 被複製
RUN echo "=== DEBUG: vendor/a2ui contents ===" && \
    ls -la vendor/a2ui/ || echo "vendor/a2ui NOT FOUND"

# 第二次安裝 (確保 workspace 連結正確)
RUN pnpm install

# 1. 編譯後端
RUN pnpm build

# 2. [關鍵修正] 顯式編譯 UI 前端
# 這一步會產生 dist/ui 或 ui/dist 資料夾，解決 "Control UI assets not found" 錯誤
RUN pnpm ui:build

# 權限修正
RUN mkdir -p /home/node/.openclaw/workspace && \
    mkdir -p /home/node/.config/gog && \
    mkdir -p /home/node/.npm && \
    chown -R node:node /app && \
    chown -R node:node /home/node && \
    chmod -R 777 /home/node