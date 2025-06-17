---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---


{% include header.html %}

---

{% include ai-share.html %}

---

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**Blog**：[2025年06月07日，用 Cloudflared 實作 SSH / HTTP / RDP Tunnel](https://blog.twman.org/2025/06/zero-trust-genai.html)

---

# 用 Cloudflared 實作 SSH / HTTP / RDP Tunnel
_Cloudflared Tunnel：不裸奔全面穿透 HTTP、SSH、RDP_

傳統雲端主機的遠端連線方式，如開啟 GCP 公網固定 IP 並設防火牆 port（如 22、3389、443 等），雖然快速直接，但也潛藏著不少諸如被掃 port、暴力破解、VPN 管理不易、身份控管與審計困難等風險。隨著資安攻擊手法日益進化，Zero Trust (零信任) 架構逐漸成為企業資安標準，其核心理念是「永不信任，持續驗證」：不論內外部網路來源，都必須經過身份驗證與存取政策評估才能進入系統。

記得好一陣子之前有做過這樣的一篇記錄，使用方法很簡單，但有流量限制。
在 虛擬機或者docker裡用 ngrok 穿透到本機 flask 的 nginx 跑 SSL WEB 或者 jupyter

```bash
# curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
| sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
| sudo tee /etc/apt/sources.list.d/ngrok.list \
&& sudo apt update \
&& sudo apt install ngrok
# ngrok http 11434 --host-header="localhost:11434"
```

而本文將示範如何透過 Cloudflared Tunnel + Cloudflare Zero Trust，在不開放 GCP 公網固定 IP 的前提下，實現對 SSH、HTTP 與 RDP 主機的安全遠端連線；按慣例，先來個表格比對：

<p align="center">
  <img src="https://github.com/Deep-Learning-101/deep-learning-101.github.io/blob/main/img/tunnel-001.jpg?raw=true" alt="Deep Learning 101">
</p>

整體方案具備以下基本特性：
*   ✅ 不開 Port、不用 VPN、不暴露主機，降低暴露風險
*   ✅ 支援 SSO、MFA，強化身份驗證
*   ✅ 審計與存取記錄集中化，方便追蹤與合規
*   ✅ 彈性政策控管，例如限制特定群組才能登入 SSH 或 RDP

## 1️⃣ 零信任為什麼重要？尤其在 AI 應用場景

大模型會「多問」、「亂問」、「記住」：比人更難控管
企業內部資源不應再預設信任任何網段或工具
Zero Trust 的四個核心：身份驗證、最小權限、動態評估、全程審計

先說在前，這需有自己的網域跟設定 DNS；如果沒有，那就忍耐用下面動態網址吧
```bash
# wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O cloudflared.deb
# sudo dpkg -i cloudflared.deb
# cloudflared tunnel --url http://localhost:80
```

有自己網域且能自己設定DNS (Ubuntu)
```bash
# Add cloudflare gpg key，先把 cloudflare的gpg key 安裝
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

# Add this repo to your apt repositories，把 repo 加入
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list

# install cloudflared，直接安裝
sudo apt-get update && sudo apt-get install cloudflared
```

有自己網域且能自己設定DNS (CentOS)
```bash
#直接下載，不用 gpgkey 方式下載安裝
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -O cloudflared.rpm

# 可能會碰上CentOS 7 已進入 EOL（終止支援） 的狀況，造成預設 yum repository 鏡像站已失效（404 或 503），導致 yum 嘗試從無效的源下載 metadata 檔案。 這 與 cloudflared rpm 本身無關，而是整個 yum 系統快掛掉了 😅
sudo cp -a /etc/yum.repos.d /etc/yum.repos.d.bak
sudo yum clean all
sudo yum makecache

sudo yum localinstall --nogpgcheck cloudflared.rpm -y

# 可能會找不到執行路徑
sudo ln -s /usr/bin/cloudflared /usr/local/bin/cloudflared

cloudflared --version
cloudflared version 2025.6.0 (built 2025-06-11-1108 UTC)
```

## 2️⃣ Cloudflared Tunnel 是什麼？如何協助實踐 Zero Trust

*   簡介架構：Tunnel client -> Cloudflare Edge -> Access 控管 -> 後端服務
*   不需開 Port，不暴露主機，支援 SSO、MFA、IP 控制
*   實際支援 HTTP / SSH / RDP 等服務，如下：

<p align="center">
  <img src="https://github.com/Deep-Learning-101/deep-learning-101.github.io/blob/main/img/tunnel-002.jpg?raw=true" alt="Deep Learning 101">
</p>

接著就是要登入，然後創建相關設定檔，我是在主機端設定，也可以在網頁端設定就是
```bash
cloudflared login
cloudflared tunnel create XXXXX
```
沒意外的話這時會取得 `xxxxx.pem` 還有 一串字串的 tunnel ID 的 json 檔
`credentials-file: /home/user/.cloudflared/XXXX-XXXX-XXXX-XXXX-XXXX.json`

再來就是要陸續針對各個服務設定，把以下兩行先寫到 `/home/user/.cloudflared/config` 裡
```yaml
# 這行指定了 Tunnel 的 UUID，
tunnel: XXXX-XXXX-XXXX-XXXX-XXXX
credentials-file: /home/user/.cloudflared/XXXX-XXXX-XXXX-XXXX-XXXX.json
```

## 3️⃣ 🔧 Cloudflared Tunnel 實作教學 ▶️ SSH 遠端管理

*   使用 `cloudflared access ssh` 配置
*   客戶端如何登入（ssh config / `cloudflared ssh`）

🔸 SSH：Cloudflared 支援 SSH Proxy，讓你不用開啟 22 port，也能安全連進主機。這同時支援使用 Cloudflare Access 進行身份驗證（如 Google Workspace、Okta 等），完全不依賴公網 IP。

> Cloudflared allows secure SSH access without opening port 22, by tunneling traffic through Cloudflare and enforcing Access policies (SSO, MFA).

*   ✅ 傳統方式：可直接連線、支援 SCP、SFTP 等。
*   🚧 Cloudflared Tunnel：需透過 `cloudflared access ssh` 或將 SSH 封裝為 HTTPS proxy（較複雜，但支援 Cloudflare Access 控制）。

`/home/user/.cloudflared/config` 裡除了 tunnel ID 跟 credentials-file 還要加上這樣，xxx就是你的子網域

```yaml
ingress:
    - hostname: xxx.twman.org
      service: ssh://localhost:22
```

同時也要在 terminal 下這樣的指令，第一個 xxxxx 就是前述的 `cloudflared tunnel create XXXXX`，第二個 xxx 就是你的子網域，就是前述的 `- hostname: xxx.twman.org` 到這都是在欲做為 tunnel 主機的設定

```bash
cloudflared tunnel route dns xxxxx xxx.twman.org
cloudflared tunnel run xxxxx
```
接著是要從你要連線至這主機的windows等機器上執行

```bash
cloudflared access tcp --hostname xxx.twman.org --url localhost:22222
```
會出現像這樣
```
2025-06-04T05:54:02Z INF Start Websocket listener host=localhost:22222
```

這時你就能從 vscode 的 remote ssh 連線到 `localhost:22222` 然後其實是連到你設定的 service 了

## 4️⃣ 🔧 Cloudflared Tunnel 實作教學 ▶️ HTTP 服務（網站 / API）

*   設定 `cloudflared tunnel`
*   設定 ingress rules
*   搭配 Cloudflare Access 做身分驗證

🔸 HTTP/HTTPS：假如你有一個內部網站或 API，不想開公網 IP，可透過 Cloudflared 將它暴露給授權用戶。

> For internal web apps (e.g., dashboards, APIs), expose them securely using Cloudflared with Access.

*   ✅ Cloudflared Tunnel 是理想方案，可結合 Cloudflare Access（MFA、SSO、IP allowlist）。
    且支援原始 IP 保留、WAF、防 bot 等功能。

http的設定也差不多，就是 hostname 跟 service 修改一下
```yaml
ingress:
    - hostname: xxx.twman.org
      service: http://localhost:80
```

一樣也要在 terminal 下這樣的指令，第一個 xxx 就是前述的 `cloudflared tunnel create XXXXX`，第二個 xxx 就是你的子網域，就是前述的 `- hostname: xxx.twman.org` 到這都是在欲做為 tunnel 主機的設定

```bash
cloudflared tunnel route dns xxxxx xxx.twman.org
cloudflared tunnel run xxxxx
```

網頁不用像 ssh 還要在要連線的本機端執行，只要 DNS 生效，基本就可以使用了 !

## 5️⃣🔧 Cloudflared Tunnel 實作教學 ▶️ RDP 遠端桌面

*   對 Windows 機器設定 Cloudflared + RDP proxy
*   可搭配 Cloudflare App Launcher 提供快速入口

🔸 RDP：RDP 通常風險更高，但 Cloudflared 支援將 RDP 流量安全地轉送。

> For Windows RDP access, tunnel port 3389 securely via Cloudflare Tunnel.

*   🚧 使用 Cloudflared 時，無法保留原始 IP。
    可以結合 Access 設定 SSO、MFA 控制入口，但仍無法看到用戶真實 IP。
    若需審計身份，推薦搭配 Cloudflare Access 或 Gateway。

這個要稍稍注意一下，看你是要轉發到那台機器的 RDP
```yaml
ingress:
    - hostname: xxx.twman.org
      service: rdp://xxx.xxx.xxx.xx:3389
```

一樣也要在 terminal 下這樣的指令，第一個 xxx 就是前述的 `cloudflared tunnel create XXXXX`，第二個 xxx 就是你的子網域，就是前述的 `- hostname: xxx.twman.org` 到這都是在欲做為 tunnel 主機的設定

```bash
cloudflared tunnel route dns xxxxx xxx.twman.org
cloudflared tunnel run xxxxx
```

接著是要從你要連線至這主機的windows等機器上執行

```bash
cloudflared access tcp --hostname xxx.twman.org --url localhost:13389
```
會出現像這樣
```
2025-06-04T05:54:02Z INF Start Websocket listener host=localhost:13389
```

這時你就能用遠端桌面連線連到 `localhost:13389` 然後其實是連到你設定的 service 了

*   ✅ 不開公網 IP，讓服務「不可見於公開網路」——這是 Zero Trust 的第一步。
*   ✅ 經由 Cloudflare Tunnel 中繼所有連線，控制入口點。
*   ✅ 加上 Cloudflare Zero Trust，可以限制誰能用 SSH / RDP / HTTP 登入，並整合 SSO、MFA。
*   ✅ 可記錄所有連線行為（審計）、控制不同帳號不同權限（最小權限原則）。

## 6️⃣💡 這些設定如何支撐 AI 應用的安全

*   將 AI 工具接入內部資料時，仍經過 Access 驗證與稽核
*   可用 Zero Trust 控管 AI Agent、plugin 或用戶行為
*   未來與 RAG、內部 API 查詢、資料倉接入等都能建立防線

## 7️⃣✨ 小結與實務建議

*   Cloudflared 是快速實現 Zero Trust 的實用工具
*   適合中小型企業或大型企業中的獨立專案團隊部署
*   若未來你要開放 AI 工具存取資料，這些設計將成為關鍵防線
