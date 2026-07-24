---
layout: default
title: "2026 內網穿透神器：Cloudflare Tunnel 免公網 IP 架設教學 (SSH/HTTP/RDP)"
description: "沒有固定 IP 怎麼辦？完整教學帶你用 Cloudflare Tunnel 實作 Zero Trust 零信任架構，3 分鐘安全架設 SSH、HTTP 與遠端桌面 (RDP) 穿透，告別防火牆設定煩惱。"
permalink: /Blog/Cloudflared-Tunnel
lang: zh-Hant
keywords: ["Cloudflare Tunnel", "內網穿透", "免公網 IP", "Zero Trust", "SSH Tunnel", "RDP", "反向隧道"]
---


{% include header.html %}

---

{% include ai-share.html %}

---

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**Blog**：[2026年03月19日更新，用 Cloudflared 實作 SSH / HTTP / RDP Tunnel](https://blog.twman.org/2025/06/zero-trust-genai.html)

---

# 《Cloudflare Tunnel 教學：免公網 IP，3分鐘架設內網穿透 (SSH/HTTP/RDP)》
_用 Cloudflared 實作 SSH / HTTP / RDP Tunnel：不裸奔全面穿透 HTTP、SSH、RDP_

> 📌 **技術速覽**
> 在免公網 IP 與無固定 IP 環境下，**Deep Learning 101** 教學展示如何透過 Cloudflare Tunnel 實作 Zero Trust 零信任架構。3 分鐘內即可建構安全的 SSH、HTTP 與遠端桌面 (RDP) 隧道，為企業內部 AI 工具與 RAG 系統打通安全的內網穿透防線。  
> 無需設定防火牆 Port Forwarding 或購買固定 IP，透過 **Cloudflare Tunnel** 即可實現安全的內網穿透。  
> 本教學詳細解說如何配置 **SSH**、**HTTP** 及 **Windows RDP** 的遠端連線，並結合 Zero Trust 驗證機制保護企業資產。  

🎯 決策者思維： 面對層出不窮的 AI 新框架，企業盲目跟風往往只會帶來高昂的試錯成本。如何跳出技術焦慮，從商業本質制定 AI 落地架構？請參考這篇策略分析：[AI 新賽局：企業導入生成式 AI 的入門策略與藍圖指南](https://deep-learning-101.github.io/Blog/AIBeginner).

🔒 企業級資安延伸： Cloudflared Tunnel 解決了網絡層的邊界安全，但如果你架設的是企業內部 AI 服務，更需要解決應用層的「輸入輸出安全檢查」。完整架構請參考：[🛡️ AI 大模型安全護欄（LLM-Guard）綜合報告](https://deep-learning-101.github.io/cyber/LLM-Guard).

🤖 負責任 AI 治理： 安全的網絡通道是企業資安的基石。在架設、開放各類內部 AI 工具的同時，如何建立完善的負責任 AI 審查機制與資料稽核治理？請參考：[🤖 企業級 AI 標竿分析與負責任 AI 治理建議報告](https://deep-learning-101.github.io/Blog/AI-Govs).

💡 進階實戰： 如果你受夠了開源 Agent 框架繁瑣的配置與高幻覺率，想體驗目前地表最強、真正由 Anthropic 原生驅動的 CLI 自動化 AI Agent 開發工具，強烈推薦閱讀：[2026 Claude Code 完全整合指南與實戰避坑](https://deep-learning-101.github.io/Blog/Claude-Code).

## 📑 目錄 (Table of Contents)
* [1️⃣ 零信任為什麼重要？尤其在 AI 應用場景](#zero-trust)
* [2️⃣ Cloudflared Tunnel 是什麼？如何協助實踐 Zero Trust](#what-is-tunnel)
  * [🌟 加碼實戰：將 Cloudflared 註冊為系統背景服務 (開機自動啟動)](#system-service)
* [3️⃣ 🔧 Cloudflared Tunnel 實作教學 ▶️ SSH 遠端管理](#ssh-tunnel)
* [4️⃣ 🔧 Cloudflared Tunnel 實作教學 ▶️ HTTP 服務（網站 / API）](#http-tunnel)
* [5️⃣ 🔧 Cloudflared Tunnel 實作教學 ▶️ RDP 遠端桌面 (含無固定 IP 解決方案)](#rdp-tunnel)
  * [🚧 如果該Windows機器沒有對外的固定IP要怎辦？ ▶️ SSH 反向隧道（Reverse SSH Tunnel）](#reverse-ssh-tunnel)
* [6️⃣ 💡 這些設定如何支撐 AI 應用的安全](#ai-security)
* [7️⃣ ✨ 小結與實務建議](#conclusion)

傳統雲端主機的遠端連線方式，如開啟 GCP 公網固定 IP 並設防火牆 port（如 22、3389、443 等），雖然快速直接，但也潛藏著不少諸如被掃 port、暴力破解、VPN 管理不易、身份控管與審計困難等風險。隨著資安攻擊手法日益進化，Zero Trust (零信任) 架構逐漸成為企業資安標準，其核心理念是「永不信任，持續驗證」：不論內外部網路來源，都必須經過身份驗證與存取政策評估才能進入系統。

記得好一陣子之前有做過這樣的一篇記錄，使用方法很簡單，去官網註冊跟取得key後，就能直接用，但有 1GB 流量限制。
在 虛擬機或者docker裡用 ngrok 穿透到本機 flask 的 nginx 跑 SSL WEB 或者 jupyter。

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
| sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
| sudo tee /etc/apt/sources.list.d/ngrok.list \
&& sudo apt update \
&& sudo apt install ngrok

ngrok http 11434 --host-header="localhost:11434"
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

<a id="zero-trust"></a>

## 1️⃣ 零信任為什麼重要？尤其在 AI 應用場景

大模型會「多問」、「亂問」、「記住」：比人更難控管；企業內部資源不應再預設信任任何網段或工具，Zero Trust 的四個核心：身份驗證、最小權限、動態評估、全程審計。

先說在前，這需有自己的網域跟設定 DNS；如果沒有，那就忍耐使用下面的快速Tunnel動態穿透網址吧
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O cloudflared.deb

sudo dpkg -i cloudflared.deb

cloudflared tunnel --url http://localhost:80
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

<a id="what-is-tunnel"></a>

## 2️⃣ Cloudflared Tunnel 是什麼？如何協助實踐 Zero Trust

*   簡介架構：Tunnel client -> Cloudflare Edge -> Access 控管 -> 後端服務
*   不需開 Port，不暴露主機，支援 SSO、MFA、IP 控制
*   實際支援 HTTP / SSH / RDP 等服務，如下：

<p align="center">
  <img src="https://github.com/Deep-Learning-101/deep-learning-101.github.io/blob/main/img/tunnel-002.jpg?raw=true" alt="Deep Learning 101">
</p>

接著就是要登入，然後創建相關設定檔，我是在主機端設定，也可以在網頁端設定就是；XXXXX 就是你要幫這個 Tunnel 取的名字
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

<a id="system-service"></a>

## 🌟 加碼實戰：將 Cloudflared 註冊為系統背景服務 (開機自動啟動)

在這篇教學文中，我們都是使用 `cloudflared tunnel run xxxxx` 來啟動隧道。但你會發現，**只要關閉終端機（Terminal），隧道就會斷線**！這在正式環境是不被允許的。

為了解決這個問題，我們必須將 Cloudflared 註冊為 Linux 的背景系統服務（Systemd Service），讓它不僅能常駐於背景，還能在主機重啟時自動連線。

> **⚠️ 重要觀念（路徑陷阱）：** > 當我們手動執行時，設定檔是讀取 `/home/user/.cloudflared/`。
> 但當轉換為「系統服務」時，程式會以 root 權限執行，並**強制改為讀取 `/etc/cloudflared/` 目錄下的設定檔**。因此我們必須幫設定檔「搬家」。

**🔧 步驟 1：建立系統資料夾並複製設定檔**
先按 `Ctrl + C` 停止你目前正在跑的 tunnel，然後執行以下指令，將憑證與設定檔搬移到系統目錄：

```bash
# 建立系統服務預設讀取的目錄
sudo mkdir -p /etc/cloudflared

# 將憑證、密鑰 (json) 與設定檔全部複製過去 (注意：請將 /home/user 替換為你的實際家目錄)
sudo cp /home/user/.cloudflared/* /etc/cloudflared/
```

**🔧 步驟 2：修改 `/etc/cloudflared/config.yml` 裡的路徑**
這步非常關鍵！請編輯你剛複製過去的 config 檔：
```bash
sudo nano /etc/cloudflared/config.yml
```
把裡面的 `credentials-file` 路徑，從原本的使用者家目錄改為 `/etc/cloudflared/`：
```yaml
tunnel: XXXX-XXXX-XXXX-XXXX-XXXX
# 記得把這行改成 /etc/cloudflared/ 開頭
credentials-file: /etc/cloudflared/XXXX-XXXX-XXXX-XXXX-XXXX.json

ingress:
    - hostname: xxx.twman.org
      service: http://localhost:80
```

**🔧 步驟 3：安裝並啟動服務**
確認設定檔都就位後，執行以下安裝指令：

```bash
# 安裝系統服務
sudo cloudflared service install

# 啟動服務
sudo systemctl start cloudflared

# 設定開機自動啟動
sudo systemctl enable cloudflared
```

最後，你可以用 `sudo systemctl status cloudflared` 來檢查狀態。如果看到綠色的 `active (running)`，恭喜你！你的 Zero Trust 隧道已經無堅不摧，就算重開機也會自動連線了！

最後，這邊再補上也同步啟動走 Tunnel 的 Jupyter 吧？

```
which jupyter
```

首先確定你的 jupyter 在那 ?

```
sudo vi /etc/systemd/system/jupyter.service
```

然後編輯其內容

```
[Unit]
Description=Jupyter Notebook/Lab Service
After=network.target

[Service]
Type=simple
# 1. 改成你平常登入 Ubuntu 的帳號名稱（例如 ubuntu 或 pi，絕對不要寫 root）
User=你的使用者名稱

# 2. 改成你希望 Jupyter 啟動時的預設資料夾路徑（例如你的專案目錄）
WorkingDirectory=/home/你的使用者名稱/Projects

# 3. 把剛剛 which jupyter 查到的路徑貼到這裡，後面接上啟動參數
# 如果你習慣用 Jupyter Lab，就把 notebook 改成 lab
ExecStart=/這裡填入/which/查到的/jupyter/路徑 notebook --no-browser --ip=127.0.0.1 --port=8888

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

接著是重新載入與啟動服務

```
# 重新載入 systemd 設定
sudo systemctl daemon-reload

# 啟動 Jupyter 服務
sudo systemctl start jupyter

# 設定開機自動啟動
sudo systemctl enable jupyter

# 確認 Jupyter 是否乖乖運作中
sudo systemctl status jupyter
```

<a id="ssh-tunnel"></a>

## 3️⃣ 🔧 Cloudflared Tunnel 實作教學 ▶️ SSH 遠端管理

*   使用 `cloudflared access ssh` 配置
*   客戶端如何登入（ssh config / `cloudflared ssh`）

🔸 SSH：Cloudflared 支援 SSH Proxy，讓你不用開啟 22 port，也能安全連進主機。這同時支援使用 Cloudflare Access 進行身份驗證（如 Google Workspace、Okta 等），完全不依賴公網 IP。

> Cloudflared allows secure SSH access without opening port 22, by tunneling traffic through Cloudflare and enforcing Access policies (SSO, MFA).

*   ✅ 傳統方式：可直接連線、支援 SCP、SFTP 等。
*   🚧 Cloudflared Tunnel：需透過 `cloudflared access ssh` 或將 SSH 封裝為 HTTPS proxy（較複雜，但支援 Cloudflare Access 控制）。

接續前面所做的取得 `xxxxx.pem` 跟 `/home/user/.cloudflared/XXXX-XXXX-XXXX-XXXX-XXXX.json` 後，編輯新增 `/home/user/.cloudflared/config` ，裡面除了 tunnel ID 跟 credentials-file 還要加上這樣，xxx就是你的子網域

```yaml
ingress:
    - hostname: xxx.twman.org
      service: ssh://localhost:22
```

同時再接著在 terminal 下這樣的指令，第一個 xxxxx 就是前述的 `cloudflared tunnel create XXXXX`，第二個 xxx 就是你的子網域，就是前述的 `- hostname: xxx.twman.org` 到這都是在欲做為 tunnel 主機的設定

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


<a id="http-tunnel"></a>

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

<a id="rdp-tunnel"></a>

## 5️⃣🔧 Cloudflared Tunnel 實作教學 ▶️ RDP 遠端桌面

*   對 Windows 機器設定 Cloudflared + RDP proxy
*   可搭配 Cloudflare App Launcher 提供快速入口
*   **適用於該Windows機器有對外的固定IP**

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

<a id="reverse-ssh-tunnel"></a>

🚧 但是就是這個BUT，**如果該Windows機器沒有對外的固定IP要怎辦？**，這時就得用上 **SSH 反向隧道（Reverse SSH Tunnel）**

*   🔧 步驟 1：下載 [plink.exe](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html#:~:text=plink.exe%20(a%20command%2Dline%20interface%20to%20the%20PuTTY%20back%20ends))

*   🔧 步驟 2：如果不確定自己 key 跑那去，可以在 Ubuntu 上執行 `ssh-keygen -t rsa -b 2048 -f ~/.ssh/rdp_key` 取得
    *   ~/.ssh/rdp_key (私鑰)
    *   ~/.ssh/rdp_key.pub（公鑰）
    *   `cat ~/.ssh/rdp_key.pub >> ~/.ssh/authorized_keys`
    *   `chmod 600 ~/.ssh/authorized_keys`
    *   `puttygen ~/.ssh/rdp_key -o ~/rdp_key.ppk` 把這個 ppk 放到 Windows 機器上

* 🔧 **步驟 3：在 Windows 機器上建立「自動重連」的雙通道 SSH 反向隧道**
    傳統直接執行 `plink` 遇到網路閃斷就會永久斷線。建議寫成 `.bat` 批次檔，加入「無限迴圈」與「斷線倒數重連」機制。
    另外，`plink` 支援一次轉發多個 Port！範例只轉發 RDP (3389)：

    請建立一個 `AutoTunnel.bat` 檔案，貼入以下內容：
    ```bat
    @echo off
    :loop
    echo [%date% %time%] 正在建立 SSH 隧道...
    "C:\plink.exe" -ssh ubuntu@your-ubuntu-ip -i C:\rdp_key.ppk -N -batch -R 0.0.0.0:3390:localhost:3389

    echo [%date% %time%] 隧道已斷線！等待 10 秒後重新連線...
    REM 利用 ping 本機來製造約 10 秒的穩定延遲 (適用於各版本 Windows)
    ping 127.0.0.1 -n 11 > nul
    goto loop
    ```
    > **💡 進階技巧 (隱藏黑視窗)**：可將此 `.bat` 檔丟入 Windows 內建的「工作排程器 (Task Scheduler)」，觸發條件設為「當電腦啟動時」，並勾選「不論使用者登入與否均執行」。這樣隧道就會變成打死不退的隱藏背景服務！

* 🔧 **步驟 4：修改 Ubuntu 上的 SSH server 配置 `/etc/ssh/sshd_config`**
    * 找到或新增這行 `GatewayPorts yes`
    * 如果你有加開 KMS 轉發，記得在 Ubuntu 防火牆放行：`sudo ufw allow 1688/tcp`
    * 重新啟動 `sudo systemctl restart sshd`
    *   這時，Ubuntu 上執行 `sudo netstat -tnlp | grep 3390`
    *   應該就能看到已監聽所有介面  
    `tcp        0      0 0.0.0.0:3390           0.0.0.0:*               LISTEN      xxxx/sshd: ubuntu`

如果你只是想快速「從外網 RDP 連進無公網 IP 的 Windows」，或需要代理內網的 KMS 認證服務，這個強化的 **SSH 反向隧道（Reverse SSH Tunnel）** 方案極度穩定且實用。
但如果環境允許安裝新版系統，利用 Cloudflare Access、Zero Trust 等功能搭配 Cloudflare Tunnel 依然是管理上最安全完整的最終目標。

<a id="ai-security"></a>

## 6️⃣💡 這些設定如何支撐 AI 應用的安全

*   將 AI 工具接入內部資料時，仍經過 Access 驗證與稽核
*   可用 Zero Trust 控管 AI Agent、plugin 或用戶行為
*   未來與 RAG、內部 API 查詢、資料倉接入等都能建立防線

<a id="conclusion"></a>

## 7️⃣✨ 小結與實務建議

*   Cloudflared 是快速實現 Zero Trust 的實用工具
*   適合中小型企業或大型企業中的獨立專案團隊部署
*   若未來你要開放 AI 工具存取資料，這些設計將成為關鍵防線

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "TechArticle",
      "mainEntityOfPage": {
        "@type": "WebPage",
        "@id": "https://deep-learning-101.github.io/Blog/Cloudflared-Tunnel"
      },
      "headline": "Cloudflare Tunnel 免公網 IP 架設教學：實作 Zero Trust 內網穿透 (SSH/HTTP/RDP)",
      "description": "無固定 IP 也能實作零信任架構！完整教學帶你用 Cloudflare Tunnel 3 分鐘安全架設 SSH、HTTP 與遠端桌面穿透。",
      "image": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg",
      "author": {
        "@type": "Person",
        "name": "TonTon Huang Ph.D.",
        "url": "https://twman.org/"
      },
      "publisher": {
        "@type": "Organization",
        "name": "Deep Learning 101, Taiwan",
        "url": "https://deep-learning-101.github.io/"
      }
    },
    {
      "@type": "FAQPage",
      "mainEntity": [
        {
          "@type": "Question",
          "name": "在無公網 IP 環境下，如何安全地開放地端 AI 工具給外部連線？",
          "acceptedAnswer": {
            "@type": "Answer",
            "text": "透過部署 Cloudflare Tunnel 搭配 Cloudflare Access，無需開放防火牆 Port，即可實現零信任 (Zero Trust) 級別的加密內網穿透與身分驗證。"
          }
        }
      ]
    }
  ]
}
</script>