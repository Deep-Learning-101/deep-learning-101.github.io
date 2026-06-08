---
layout: default
title: "2026 Claude Code 完全整合指南：從核心機制、工程紀律、大規模部署到生態擴展"
description: "深度整合了多篇Claude Code核心文章與開源項目，旨在提供一套可查閱、可執行、有層次的全景知識體系，而非功能羅列。"
permalink: /Blog/ClaudeCode
lang: zh-Hant
keywords: ["Claude Code", "AI Agent", "CLI", "Developer Tools", "Anthropic", "OPUS / SONNET / HAIKU"]
---


{% include header.html %}

---

{% include ai-share.html %}

---

**作者**：[TonTon Huang Ph.D.](https://twman.org)  

---

# Claude Code 完全整合指南：從核心機制、工程紀律、大規模部署到生態擴展

> **定位**：整合了Claude Code核心文章與開源項目，以及自己的使用體驗，目的是提供一套可查閱、可執行、有層次的全景知識體系，而非功能羅列。

---

## 目錄（點擊跳轉）

- [1. Claude Code 是什麼？為什麼它「感覺不同」](#1-claude-code-是什麼為什麼它感覺不同)  
- [2. 與 Copilot, Cursor, Antigravity的差異](#2-與-copilot--cursor--Antigravity的差異)  
- [3. 安裝與起步](#3-安裝與起步)
- [4. 核心機制：CLAUDE.md —— 讓 AI 真正懂你的專案](#4-核心機制claudemd--讓-ai-真正懂你的專案)
- [5. Plan Mode：先想清楚再動手](#5-plan-mode-先想清楚再動手)
- [6. 上下文管理：這份工具的命脈](#6-上下文管理-這份工具的命脈)
- [7. 工作流哲學：給驗證標準、四階段循環、怎麼跟 Claude 溝通](#7-工作流哲學-給驗證標準四階段循環怎麼跟-claude-溝通)
- [8. Skills 深度整合：工程紀律的「四層架構」](#8-skills-深度整合-工程紀律的四層架構)
- [9. Skills 安裝、管理與優先級建議](#9-skills-安裝管理與優先級建議)
- [10. 大規模代碼庫與企業級框架：Harness 比模型更重要](#10-大規模代碼庫與企業級框架-harness-比模型更重要)
- [11. 避坑指南：失敗模式對照表](#11-避坑指南-失敗模式對照表)
- [12. 生態擴展一：free-claude-code —— Claude Code 的「換引擎」本地閘道](#12-生態擴展一free-claude-code--claude-code-的換引擎本地閘道)
- [13. 生態擴展二：cc-connect —— 把本地 Agent 橋接到你的聊天平台](#13-生態擴展二cc-connect--把本地-agent-橋接到你的聊天平台)
- [14. 附錄：指令 / 命令速查表](#14-附錄-指令命令速查表)

---

<a id="1-claude-code-是什麼為什麼它感覺不同"></a>
## 1. Claude Code 是什麼？為什麼它「感覺不同」

**一句話定調**：

> **「其他 AI 工具是幫你寫代碼，Claude Code 是替你執行任務。」**  
> Skills 不是讓它變聰明，Skills 讓它變**自律**。

它是 **Anthropic 推出的終端原生 AI Agent（代理）**，能：

-   讀懂「整個專案」結構（跨文件、跨層級推理）
-   自主規劃 → 修改多個文件 → 執行 shell / test / build
-   透過 **`CLAUDE.md`** 學你的專案規範，把「專案知識」變成長效上下文
-   把很多原本要開幾十個 tab、反覆試錯的過程，**坍縮成一個對話式工作流**
-   在 SWE-bench 測試中，其解決真實 Issue 的能力獲得了 80.8% 的驚人成績。  

這種體驗有一個很準的詞：**認知壓縮（Cognitive Compression）**。

> 它最關鍵的差異不在「生成更準」，而在於它理解**系統**而非片段：重構時是真的在重組架構邏輯，debug 時會推理 Bug 擴散鏈，接手陌生 repo 時能跨文件連概念，而不只是孤立解釋「這個函式做什麼」。



**目標受眾分析**：

* ✅ **極度適合：** 具備開發基礎想放大產出的工程師、需要經常寫腳本處理資料的 PM/營運、想邊做邊學的技術創業者。
* ⛔ **不適合：** 完全零基礎且沒用過終端機的人（建議先用網頁版）、企業內網完全無法連外網的封閉場景。

---

<a id="2-與-copilot--cursor--Antigravity的差異"></a>
## 2. 與 Copilot / Cursor / Antigravity的本質差異

### 六大 AI 編程工具核心差異對比（更新台灣用語版）

| 工具 | 定位 | 核心工作模式 | 上下文 | 核心優勢 / 使用場景 |
| :--- | :--- | :--- | :--- | :--- |
| **GitHub Copilot** | 智慧程式碼補全 | 整合在 IDE 中，**即時**提供單行或多行程式碼建議。 | ~8K tokens | **補全速度極快**，無縫融入既有的編碼流程，**幾乎沒有學習成本**。 |
| **Cursor** | AI 增強型 IDE | 以 VS Code 為基礎深度整合 AI（聊天與 **Composer** 模式），支援多檔案編輯與對話。 | 專案感知 | 保留完整的 **VS Code 體驗**，同時提供強大的聊天、編輯與**內建代理**功能，生態系完整。 |
| **Claude Code** | 終端原生 Agent | 在終端機裡作為**自主代理**執行，能夠讀取檔案、執行指令、規劃任務。 | 200K tokens | 具備**系統級推理**與**自主任務執行**能力。擅長複雜重構、偵錯、遷移等需要跨檔案、多階段操作的專案級任務。 |
| **Google Antigravity** | **智慧型體優先 (Agent-first) IDE** | 基於 VS Code 建構的整合開發環境，可將複雜編碼任務**委託**給由 Gemini 3 Pro 等模型驅動的自主 AI 智慧型體執行。智慧型體能直接操作編輯器、終端機與瀏覽器。 | 取決於所選模型（支援 Gemini 3 Pro, Claude Sonnet 4.5, OpenAI 模型等） | **實現「智慧型體優先」的高自主性開發**。智慧型體會產生可檢驗的「工作成品」（如任務清單、實作計畫、截圖），而非僅是原始工具呼叫，旨在建立使用者信任並完成從規劃到實作的完整任務鏈。 |

---

### 如何選擇與搭配使用？

基於以上更完整的工具生態，一個有效率的現代化開發工作流可以這樣安排：

1.  **日常編碼（在編輯器內）**：
    *   **GitHub Copilot**：用於**毫秒級**的程式碼片段自動完成，提升純打字效率。
    *   **Cursor**：作為主力 IDE，當需要主動使用其智慧聊天和重構功能時使用。

2.  **複雜任務委託（終端機或專屬 IDE）**：
    *   **Claude Code**：作為**主力工程智慧型體**。當有明確、複雜的開發任務時，在終端機中交給它來規劃與執行。
    *   **Google Antigravity**：當您希望**在一個 IDE 環境內**，以更高自主性的方式，將一個完整功能或模組的開發工作委託給 AI 智慧型體，並透過其產生的「工作成品」來追蹤與驗證執行過程時使用。

**總結**：當前的 AI 輔助編程生態已形成多層次的工具矩陣，從被動補全、主動對話，到高自主性的智慧型體執行。**Antigravity 的加入**代表了「智慧型體優先」理念在 IDE 層級的正式產品化，與終端 Agent 和傳統增強型 IDE 形成了差異化競爭。開發者可根據任務的粒度、執行環境的偏好，以及對智慧型體自主性的信任程度，靈活選用最適合的工具。

---

<a id="3-安裝與起步"></a>
## 3. 安裝與起步

### 3.1 前置條件

-   **[Claude Pro](https://claude.ai/upgrade/pro)**（$20/月）或更高（Max 5x $100 / 20x $200）
-   **Node.js ≥ v18**
-   能連 Anthropic 的網路環境

### 3.2 安裝（通用方式）

```bash
npm install -g @anthropic-ai/claude-code

claude --version
claude
```

首次啟動會引導：
1. 選主題（暗色／亮色）
2. Claude 帳號授權
3. 信任當前工作目錄

> **三個入口**：終端 CLI（最深、最專注）／桌面 App（獨立視窗）／網頁&手機（構思用）。共用同一個帳號。
> **IDE 整合補充**：除終端 CLI 外，Claude Code 亦可作為擴充套件整合至 **VS Code** 與 **JetBrains** 系列 IDE，在編輯器側邊欄直接操作；進階場景下還可掛進 **GitHub Actions**，自動執行 PR 審查或代碼生成。三種終端以外的入口共用同一帳號與 `CLAUDE.md` 規範。

### 3.3 費用與額度（實務要知道的事）

| 方案 | 價格 | 每 5hr 窗口恢復額度 | 備註 |
| :--- | :--- | :--- | :--- |
| Pro | $20/mo | 約 45 條對話級 | 多數個人夠用 |
| Max 5x | $100/mo | 約 225 | 重度日活 |
| Max 20x | $200/mo | 約 900 | 團隊級消耗 |

> 額度以「5 小時窗口」滾動計算；吃完就要等窗口重置或升級。  

> [透過 Amazon Bedrock 使用 Claude Code 參考這](https://code.claude.com/docs/zh-TW/amazon-bedrock)

---

<a id="4-核心機制claudemd--讓-ai-真正懂你的專案"></a>
## 4. 核心機制：CLAUDE.md —— 讓 AI 真正懂你的專案

### 4.1 它是什麼

放在**專案根目錄**的 `CLAUDE.md`，Claude Code 每次啟動自動讀取，相當於給專案一本「工作手冊」。

-   **拒絕百科全書：** 根目錄的 `CLAUDE.md` 應極度精簡（建議控制在 50 行以內），只放最高層級的指標與絕對禁止的陷阱。
-   **分層載入（Additive Loading）：** 採用目錄樹結構。例如，開發包含前端與資料庫的專案時，不要在根目錄寫滿 SQL 規範。讓 AI 在進入 `/frontend` 目錄時才讀取 Next.js 的規範（例如：核心 UI 區塊通常位於 line 104），進入 `/db` 時才讀取 PostgreSQL 與 Drizzle ORM 的設定。
-   **主動降噪：** 透過 `.ignore` 或 `.claude/settings.json` 中的 `permissions.deny` 排除生成的構建檔案（Artifacts），這能大幅節省 AI 的注意力與 Token。

### 4.2 應該寫什麼（三層原則）

**✅ 要放**（專案特有、Claude 猜不到的）：理想的配置必須涵蓋三層邏輯：  
-   1.  **What（專案用途與邊界）：** 描述技術棧與目錄結構。例如：「這是一個整合了 LINE Pay 與 GA4 事件追蹤的訂位系統，前端 UI 核心區塊位於 line 104，請優先從該處切入分析。」
-   2.  **Why（架構決策原因）：** 解釋為什麼這樣設計。例如：「我們的支付閘道採用特定的異步回調，這是為了防範 Zero-day 威脅，請勿擅自重構為同步請求。」
-   3.  **How（開發與測試規範）：** 明確指出代碼風格（如 YAML Front Matter 規範）、構建指令，以及 AI 修改代碼後必須執行的測試步驟。

**❌ 不要放**：
-   Claude 讀碼就知道的事
-   標準語言慣例
-   會頻繁變的東西
-   冗長教學文

### 4.3 最小可用模板（直接可貼）

```markdown
# CLAUDE.md — <專案名>

## 技術棧
- Next.js 14 (App Router) / Python FastAPI / …
- TypeScript / Prisma + PG / Tailwind …

## 目錄結構（一句話定位）
- app/       頁面與路由
- components/ 公共元件
- lib/        工具與 client
- prisma/     schema

## 常用命令
- dev:    npm run dev
- build:  npm run build
- test:   npm run test
- db:     npx prisma migrate dev

## 規範（只寫「你要它守的」）
- 元件用 function、async/await
- 只用 Tailwind，不寫 CSS 檔
- 每次功能開 branch；每完成一個階段就 commit
- 不得直接改 prisma/schema.prisma 除非我說
```

> 讓 Claude 幫你生初稿：`幫我分析這個專案，生成 CLAUDE.md …`，再手修。提交進 git，全團隊受益。

---

<a id="5-plan-mode-先想清楚再動手"></a>
## 5. Plan Mode：先想清楚再動手

### 5.1 怎麼進入

-   `Shift + Tab` 兩次（左下出現模式標識）
-   Windows 若不順：`Alt + M`

Plan Mode 下 Claude **只能讀文件/搜尋/問你問題**，**不能改文件、不能執行命令**。

### 5.2 什麼時候該用

-   新功能涉及 **>3 個文件**
-   動 DB schema
-   不確定方案
-   接手陌生專案

### 5.3 實戰感覺

> 你描述需求 → Claude 先問關鍵假設（「要去重嗎？用 Redis？實時還是批次？」）→ 出一份實施計畫（新增哪個欄位、改哪些檔、順序）→ 你確認 → 退出 Plan Mode → 執行

這道「強制先思考」的閘門，能把**返工機率壓低非常多**（實務感受約 ~80% 的無謂返工消失）。

---

<a id="6-上下文管理-這份工具的命脈"></a>
## 6. 上下文管理：這份工具的命脈

核心約束：**上下文視窗再大，也會滿；滿了以後性能下滑、會忘、會犯蠢**。

| 用量 | 狀態 | 建議 |
| :--- | :--- | :--- |
| 0–50% | 正常 | 自由工作 |
| 50–70% | 注意 | 避免再大量讀新文件 |
| 70–90% | 危險 | `/compact` 壓縮 |
| 90%+ | 臨界 | **`/clear` 重置**（你的檔案還在，只是清對話） |

### 6.1 關鍵命令

```text
/compact           壓縮歷史（保留關鍵資訊）
/clear             完全重置對話（CLAUDE.md 還在，專案檔不動）
/btw               快速問、答案不進歷史（省上下文）
Esc                停動作、保留上下文
Esc+Esc / /rewind  回退到某個檢查點
claude --continue  新終端續上最近 session
```

### 6.2 讓專案「自我記錄」

階段完成後，讓 Claude 更新 `CLAUDE.md` 或 `docs/`，寫三行就夠：

```markdown
<!-- docs/progress.md -->
- [x] 閱讀量：加 DB 欄位 + API 回傳
- 下一步：前端顯示、用 SWR 做 revalidate
- 已知坑：count 要避免 N+1，改用 SELECT COUNT 快取
```

下次 `/clear` 後重開，一句「我們做到 DB 欄位完成，現在要做前端顯示」就能快速恢復。

### 6.3 上下文決策框架：Every Turn Is a Branching Point

> 關鍵主張：**1M tokens 視窗再大，也只是「裝得下」，不代表「乾淨」**；差異點不在容量，而在你怎麼做 decision point。

Claude Code 的上下文視窗是 **1M tokens**，但用久了會出現 **context rot（上下文腐化）**：注意力被攤薄、早先不相關內容干擾當前目標、signal-to-noise 比下降。通常在 **300k–400k tokens** 附近就會開始有感（視任務而定，不是死閾值）。

#### 6.3.1 每一輪結束，你都站在一個分叉點（5 選 1）

Claude 完成一輪後，你的選項不是只有「繼續聊」，而是：

| 選項 | 指令/動作 | 適用 |
|---|---|---|
| **Continue（續聊）** | 直接送下一條訊息 | 同一任務、方向清楚、上下文仍乾淨 |
| **/rewind（回到分叉點重走）** | `Esc+Esc` 或 `/rewind` | 你發現它走歪了，**不要「補 patch prompt」**，而是回到還沒走歪的點再下正確約束 |
| **/clear（開新 session）** | `/clear` → 帶一則 distill 過的 brief（一句話說明現狀＋禁止事項） | 任務切換、或上下文已污染到不值得救 |
| **/compact（有損壓縮）** | `/compact`，可加 focus：`/compact focus on auth refactor, drop test debugging` | 想保留「有效約束」，但丟掉試錯廢氣 |
| **Subagent（乾淨隔離調查）** | 透過 `/dispatching-parallel-agents` 等 Skill | 下一段要掃很多檔案、產大量中間輸出，但你只需要結論 |

> ⚠️ **最重要的一句話**：  
> **Rewind 往往比「補一句糾正」更好。**  
> 因為 `/rewind` 會把那個分支之後的訊息「從上下文直接拔掉」，而補 prompt 只是在壞分支上繼續堆解釋。

實戰範例（你會懂的那種）：

- Claude 讀了五個檔、試了方案 A 失敗
- 直覺反應：「A 不行，換 B 試試」← 留在壞分支
- 更好做法：`/rewind` 回到「剛讀完檔」的節點 → 直接說：
  > 「不要用方案 A，`foo` 沒暴露那個能力，直接走 B。」

你還可以用 **summarize from here** 做「來自未來的紙條」：只保留結論，把中間的試錯丟掉。

#### 6.3.2 compact vs. /clear（lossy vs. 你重控邊界）

| | `/compact` | `/clear`＋brief |
|---|---|---|
| 本質 | **有損壓縮**：讓模型總結歷史→用摘要續行 | **你重建邊界**：你寫 brief，決定帶什麼、丟什麼 |
| 省力？ | 高（讓它幫你壓） | 低（你要寫一句現狀摘要） |
| 可控？ | 較低（它決定什麼重要） | 較高（你決定） |
| 地雷 | **bad compact**：它壓縮時往往模型狀態最差（上下文已長、rot 開始作用），而且它無法預測你下一步要往哪 | 無（但你懶就會寫太爛的 brief） |

- compact 不是「無損繼承」；它壓完建議你**花一分鐘確認摘要保留了哪些關鍵約定**。
- 更穩的作法：**趁你還知道下一步要哪，主動 compact**；別等到 autocompact 逼你吃 surprise。

#### 6.3.3 Subagent 首先是「上下文管理工具」（不是只是平行加速）

用 subagent 的判斷題：
- 這塊工作會產**大量中間輸出**（搜尋、掃描、報告）
- 你之後**只需要結論**，不需要那些 tool output 本身留在主線
- 你想避免主線上下文被淹沒

典型：
- 呼叫子代理 Skill（如 `/dispatching-parallel-agents`） → 建立唯讀 explore agent 掃清依賴/影響面 → 主 agent 才動手改
- 用 subagent 跑「依 spec 驗證」、「跨 repo 讀 auth flow」、「針對 git changes 寫 doc」

> 一句話：**Subagent 給你一個 fresh context window，中間垃圾不回傳，主線就不髒。**

### 6.4 權限與沙箱：allow / ask / deny（讓它跑，但不讓它亂跑）

Claude Code 最煩的不是「不夠強」，是**它太能幹，所以你得定邊界**。  
解法不是「關掉所有確認」（會被它摸到 `.env` 或誤 push），而是分三層：

#### 6.4.1 三層定義（`.claude/settings.json` 的 permissions）

| 層級 | 行為 | 適合放什麼 |
|---|---|---|
| **allow** | 低風險高頻：直接放行 | `git status`、`git diff`、`npm run test`、`npm run lint` |
| **ask** | 有風險但要做：保留確認 | 裝依賴、跑 build 腳本、改專案檔案 |
| **deny** | 絕不碰：直接擋 | `.env*`、金鑰路徑、生產發布腳本 |

範例片段（JSON 寫法）：

```text  
// .claude/settings.json
{
    "permissions": {
    "allow": [
        "Bash(npm run test *)",
        "Bash(npm run lint)",
        "Bash(git status)",
        "Bash(git diff *)"
        ],
// deny 優先級最高：就算別處 allow，命中 deny 還是擋
    "deny": [
        "Read(./.env)",
        "Read(./.env.*)",
        "Bash(git push *)",
        "Bash(rm -rf *)"
        ]
    }
}  
```

> ⚠️ 實務鐵律：**`.env`、金鑰目錄、發布指令優先寫進 `deny`，不要靠「我到時候看彈窗再判斷」。**

#### 6.4.2 sandbox：讓它在邊界內連續跑

`sandbox` 的用意是「設執行邊界」——專案目錄內的讀取/執行/驗證可連續跑，不每一步都來問你；但需要專案外路徑（全域工具、Docker）時要額外配。

適合場景：跑 Stripe Webhook 邏輯驗證、跑測試→看 log→檢查 DB 狀態這種「一串要連貫」的流程。

#### 6.4.3 security-review：涉使用者資料時的「安全視角」

每次涉及：
- **使用者資料查詢/存取**
- **權限檢查、外部輸入校驗**
- **支付、Webhook、檔案處理**

→ 寫完就跑，別等到要上線才跑。

最常見的兩個地雷：
1. **IDOR（越權）**：查詢只寫 `WHERE id = {userId}` 卻沒加 `AND ownerId = currentUser.id`
2. **GDPR 痕跡**：錯誤日誌把使用者 Email/IP/個資噴出來

最小作法：  
- 涉及上述場景：**寫完就跑** `/security-review`（或對應 review skill）  
- 發現問題的成本在「剛寫完」最低，拖到三週後找回來改最貴

### 6.5 Prompt Caching：大幅降低大上下文成本
對於每次呼叫都需要攜帶大量重複內容（例如龐大的 System Prompt、參考文件或整個 Repo）的場景，務必善用快取機制來控制開銷。
- **配置方式**：在需要快取的內容區塊加上 `"cache_control": {"type": "ephemeral"}`。
- **效益與生命週期**：命中快取的 token 成本最多可降九成；快取預設保留 5 分鐘，每命中一次便會重新計時。

---

<a id="7-工作流哲學-給驗證標準四階段循環怎麼跟-claude-溝通"></a>
## 7. 工作流哲學：給驗證標準、四階段循環、怎麼跟 Claude 溝通

### 7.1 最重要的一條鐵律：**永遠給成功條件**

| ❌ 模糊 | ✅ 有驗證標準 |
| :--- | :--- |
| 實作 validateEmail | 寫 validateEmail；測資：`a@b.c`→true、`invalid`→false |
| 把儀表板弄好看點 | 貼設計截圖，實作後截圖比對差異再修 |
| build 失敗 | 貼錯誤訊息；修完且驗證 `npm run build` success |

沒有驗證標準，AI 會產出「看起來對但邊界壞掉」的東西。

### 7.2 四階段循環（推薦預設）

1.  **Explore**：只讀不寫（Plan Mode），搞清楚地形
2.  **Plan**：產出有序步驟＋依賴＋待確認假設
3.  **Implement**：退出 Plan Mode → 編碼＋跑指令驗證
4.  **Commit**：`git commit` + PR；階段完成後更新 CLAUDE.md

小任務（改 typo、加一行 log）可跳過全套開銷，直接做。

### 7.3 怎麼下指令才有效

-   用 `@filename` 引用文件
-   貼 log / stack trace / 截圖
-   縮範圍：**「只改 X 檔的 Y 行為，不要動其他檔」**
-   描述「症狀」而非只丟結論：`"用戶報逾時後登入失敗，檢查 src/auth/…"`

### 7.4 把 Claude 當 senior engineer 問事

不是只叫它改；多問：
-   這個日誌系統怎麼運作的？
-   為什麼 `ExecutionFactory` 的 API 演變成這樣（叫它看 git log）？
-   如果我們加 OAuth2，你會怎麼切責任？

### 7.5 Claude Code 團隊內部的 5 條實戰原則

根據 Anthropic 內部開發團隊的工程總監分享，AI 介入後工程瓶頸已從「寫碼」轉移到「驗證與審查」，並衍生出以下五條新規範：
- **JIT (Just-In-Time) 規劃**：砍掉「先寫漂亮設計文件才能寫碼」的儀式，改為 prototype-first；文件若需要存在，是寫完代碼後再補。
- **自動化文化**：現在自動化成本極低，團隊奉行「重複 3 次以上就想辦法自動掉」（例如每天手動彙整客戶回饋）。
- **Code Review (Trust but verify)**：將 style/lint、明顯 bug 交給 Claude 處理，人類專注於法律合規、風險容忍、產品品味與信任邊界。
- **品味稀缺，打字不稀缺**：團隊角色界線模糊化，能辨識該做什麼的「創意 builder（product sense）」與深度的「系統專家」成為最需要的兩種特質。
- **主動推動變革**：人不會主動刪除流程，必須指名道姓宣告哪些舊流程（如無效率的週會或審批）可以廢除。

---

<a id="8-skills-深度整合-工程紀律的四層架構"></a>
## 8. Skills 深度整合：工程紀律的「四層架構」

### 8.1 為什麼需要 Skills（一句話）

> Skills 本質＝**把工程規範數位化**：先想、再計畫、再執行、測試覆蓋、驗證完成、乾淨分支。  
> 如同前文所述，Claude 已經具備頂級的推理智力，而 Skills 的任務，是賦予它企業級開發所需的**工程紀律**。

---

### 8.2 第一層：規劃與思考（解決「一上來就亂寫」）

| 名稱 | 它在解什麼痛點 | 實務重點 |
| :--- | :--- | :--- |
| **Brainstorming（頭腦風暴前置）** | 阻止「你描述需求→它直接開寫→方向錯→重來」 | 強制先列 3+ 實作路徑＋權衡＋風險＋要你確認的假設；適合「創造新功能、架構決策」 |
| **Writing Plans（計畫書生成）** | 大任務黑盒化→白盒化；跨 session 上下文會丟 | 輸出 5–10 步有序計畫（每步可驗證、有依賴）；搭配 `findings.md`/`task_plan.md`/`progress.md` 把記憶放磁碟 |
| **Executing Plans（按計畫執行不跑偏）** | 每步完成必驗證預期輸出，失敗就停、報問題，不允許自行改路 |
| **Planning-with-files（文件型外部記憶）** | 強制把計畫/調研/進度寫進專案目錄檔案；每次重要決定前重讀、完成後更新（目前最實的「短期記憶替代」） |

**怎麼用這一層**：把 Brainstorming 當「大決策閘門」；Writing Plans 當「複雜任務骨架」；Planning-with-files 當「跨天/跨 session 的錨」。

---

### 8.3 第二層：品質保障（讓「完成」兩個字有分量）

| 整合後名稱 | 它在解什麼痛點 | 實務重點 |
| :--- | :--- | :--- |
| **TDD Skill（測試驅動）** | AI 先寫實作→補的測試在測「自己的實作」而非業務行為 | 拒絕直接寫實作；先問行為/成功標準/邊界→先寫測資讓你確認→再實作通過測試；（Test Writer 版強調：先掃你現有測試風格再仿寫） |
| **Systematic Debugging（結構化除錯）** | 四階段鎖死：Observe→Hypothesize（3-5 假設排序）→Test（最小驗證）→Fix（根治＋防復發）；不允許在 Observe 沒做完前猜原因 |
| **Requesting Code Review（多 Agent 審查）** | 普通 AI review 變安慰式「看起來不錯」 | 召 5 個並行專項（安全/效能/正確性/風格/測試覆蓋）→匯總成 HIGH/MED/LOW 置信度報告＋檔名＋行號 |
| **Verification Before Completion（完工清單）** | Claude 的「完成」有時＝code done≠really done | 完工前強制跑核對表：test 真跑過？硬編碼抽配置？錯誤處理只有 happy path？安全掃描？README 過期？輸出 ✅❌⚠️ |

**怎麼用這一層**：
-   核心模組＝TDD＋Verification
-   線上詭異問題＝Systematic Debugging
-   PR 前＝Requesting Code Review（當你跟自己代碼「距離太近」時尤其值）

---

### 8.4 第三層：執行效率與協作流（把小時級變成分鐘級）

| 整合後名稱 | 它在解什麼痛點 | 實務重點 |
| :--- | :--- | :--- |
| **Dispatching Parallel Agents（並行分派）** | 獨立子任務卻順序跑，浪費時間 | 識別無依賴子任務→分給多 sub-agent 同時跑；每個任務描述必須**自包含**；適合「給 N 個模組加同一套入參校驗/日誌 middleware」 |
| **Using Git Worktrees（分支隔離）** | 每個分支一個隔離目錄→Claude 在裡面工作→提供 merge/PR/放棄標準流程；告別 stash 衝突焦慮 |
| **Commit Helper（提交規範）** | 分析 `git add` 的改動→自動生成 Conventional Commits 格式＋「為什麼改」（被多篇評為**CP 最高、天天用**的 skill） |
| **Batch（內建，不用裝）** | 批量改造幾十個介面/依賴，手動搞到死 | Claude Code **內建** `/batch`；拆成隔離子任務→平行跑→自動開 PR；適合統一改 response 格式、批次升依賴 |

---

### 8.5 第四層：擴展、特殊需求、團隊資產化

| 整合後名稱 | 定位 | 何時值得 |
| :--- | :--- | :--- |
| **Ralph-Wiggum（自主循環）** | Stop-hook 攔退出→把任務＋上下文重新餵回→形成 loop | 只適合「有清晰驗收標準的批量活」（加註釋、格式化等）；**必設 `--max-iterations`，否則危險** |
| **MCP Builder** | 從設計→測試的 MCP Server 開發流程 | 你要替團隊接內部 API/DB 成 MCP；一般使用者現成 MCP 就夠 |
| **Skill Creator（元技能）** | 用自然語言生 `SKILL.md`＋測試 | 價值在**團隊**：把內部規範/領域知識封裝成 Skill 推給所有人；個人用「裝現成」通常就夠 |
| **Document Skills（官方）** | Claude 在終端直接吃 PDF/Word/Excel/PPT | 非純開發者、要處理大量需求/技術文件時省事（Anthropic 官方出品） |
| **Frontend Design** | 注入設計規範讓 UI 代碼別那麼「2015 管理後台感」 | 內部系統/工具產品夠用；差距是「從隨便堆→基本規範」，不是魔法 |
| **Excalidraw Diagram** | 自然語言→可編輯 `.excalidraw` 圖 | 技術方案圖/架構圖/時序圖快速草圖；Excalidraw 可再編輯＋JSON 可存 git |

**Document Skills 四件套說明**（Anthropic 官方出品）：

| **Office 四件套 (docx/pptx/xlsx/pdf)** | 文件產出（不自己排版） | 能生成排好版的 Word、帶配色與字體的 PPT、帶公式的 Excel；PDF 支援 OCR 提取合約條款 |
| **Schedule (定時任務)** | 週期性自動化作業 | 設定好時間/週期，Claude 自動執行並推送結果（如：每週五自動生成週報） |
| **Context7** | 突破訓練資料截止日限制 | `/plugin install context7@claude-plugins-official`；拉取最新官方文件注入上下文，極度適合快迭代的前端框架 |
| **Playwright MCP** | 用自然語言測試 UI | `/plugin install playwright@claude-plugins-official`；打開真實 Chrome 點擊與截圖，適合探索性測試 |
| **Security Guidance** | 零配置的安全防線 | `/plugin install security-guidance@claude-plugins-official`；在 PreToolUse 攔截命令注入/XSS/危險 eval |
| **Chrome DevTools MCP** | 接管瀏覽器開發者工具 | 讀取網路請求、Console 報錯與 DOM；極度適合調試線上問題與登入態 Bug |
| **Figma MCP** | 設計稿直出前端代碼 | `/plugin install figma@claude-plugins-official`；讀取真實 frame 結構化資料而非截圖辨識 |

---

### 8.6 安裝來源（用一句話講清楚）

-   9 個主體來自 **superpowers 包**：
    ```
    /plugin install superpowers-skills@anthropics-claude-code
    /reload-plugins
    ```
-   另有中文社群版：`/plugin install superpowers-zh@jnMetaCode`
-   `excalidraw-diagram` 單獨：
    ```
    /plugin install excalidraw-diagram@claude-plugins-official
    ```

---

<a id="9-skills-安裝管理與優先級建議"></a>
## 9. Skills 安裝管理與優先級建議

### 9.1 建議的「上手段階梯」

1.  **先裝 Commit Helper**（天天用、CP 最高、干擾最小）
2.  **再加品質閘門**：Code Review ＋ Planning-with-files（解真痛點）
3.  **再上約束**：Brainstorming / Writing Plans / Executing Plans（接受被管，換大任務可控）
4.  直接將內建的 /batch 融入日常批量任務中
5.  **Ralph-Wiggum / MCP Builder**：等有「明確批量任務」或「要接內部系統」再碰

### 9.2 常見疑惑一次答

-   **會強制改變所有行為嗎？** 不會。多數 Skill 是「有意圖觸發」才生效；你可說「跳過 brainstorming 直接執行」。
-   **跟 Hooks 差在哪？** Skill＝方法論層（工作流程）；Hook＝事件觸發層（確定性動作，例如 commit 前必 lint）。
-   **裝太多會不會衝？** Skills 按需載入，不常駐；但不同來源的 Skill 若「都想接同一類任務」要留意優先級（superpowers 內部有元 Skill 管優先序）。


---

<a id="10-大規模代碼庫與企業級框架-harness-比模型更重要"></a>
## 10. 大規模代碼庫與企業級框架：Harness 比模型更重要

> **決定 Claude Code 在千萬行 monorepo 能走多遠的，不是模型基準，而是 Harness（生態搭建品質）**。

### 10.1 Claude 怎麼導航：Agentic Search，不是 RAG 索引

-   Claude Code 不走「embedding 索引整個 repo」那套（易過時、追不上活躍提交）
-   而是像人：遍歷檔案系統、grep、沿引用跨文件跳
-   代價：**高度依賴起手上下文品質** → 所以 CLAUDE.md、分層結構、ignore 規則才關鍵

### 10.2 Harness 的五個擴展點（重要性有順序）

| 擴展點 | 角色 | 實務重點 |
| :--- | :--- | :--- |
| **CLAUDE.md** | 第一基石（每次讀） | 精簡、分層、只放普遍適用內容 |
| **Hooks** | 事件觸發的自動化 | lint/format 這種「確定性」比靠 prompt 可靠；stop hook 可做會話後反思→建議 CLAUDE.md 更新 |
| **Skills** | 專長按需出現（漸進披露） | 可綁路徑（只在某目錄激活）；別把一切塞 CLAUDE.md |
| **Plugins** | 打包成可分發單元 | 避免好設置只活在部落；新人不需從零配 |
| **MCP Servers** | 接外部世界 | 建議最後做：基礎穩了再接內部工具/工單系統 |

除了上述五大擴展點，還有兩個極高投資報酬率的輔助機制：**LSP**（符號級導航，對 C/C++/多語大型 repo ROI 極高）與 **Subagents**（子代理探索）。

以下針對實務上最容易誤解的三個進階機制，進行深度拆解：

-   **Hooks (確定性約束與反饋迴圈)**：與其在 Prompt 裡千叮嚀萬交代「請記得做語法檢查」，不如利用 Hooks 建立強制性的工程約束。
    * **PreToolUse（事前攔截）：** 阻擋危險操作。例如，修改到特定敏感的系統配置檔前，強制觸發備份腳本。
    * **PostToolUse（事後修正）：** 修改代碼後自動觸發 Lint 或測試。如果報錯，AI 會直接讀取 stderr 並自我修正，無須人工介入。

-   **Skills (漸進式揭露的模組化能力)**：
    * **⚠️ 錯誤認知：** 許多人會把 Skills 當成零散的快捷指令，或把所有工作流硬塞進 `CLAUDE.md`。
    * **✅ 正確理解：** Skills 是一種 **「漸進式揭露 (Progressive Disclosure)」** 的機制。當你有複雜任務（如：執行安全掃描），將其打包成 `SKILL.md`。Claude 啟動時只會讀取其名稱與簡介（約 100 tokens），只有當任務匹配時，才會完整載入深層邏輯。這讓 AI 的上下文保持絕對乾淨。

-   **Subagents 與 MCP (外部探索與系統接入)**：
    * 當基礎穩固後，才透過 MCP 接入外部工具。
    * **Subagents（子代理）的防護價值：** 用來拆分「探索」與「編輯」。面對高複雜度架構，不要讓主 Agent 直接動手。先派遣一個唯讀的 Subagent 去爬梳依賴關係與圖譜，寫出調查結論後，主 Agent 再依據該結論進行精準改檔，藉此保護主 Agent 珍貴的 Context Window 不被中間的試錯過程污染。

### 10.3 三種成功配置模式（可直接抄）

1.  **讓代碼庫保持「Claude-navigable」**
    -   CLAUDE.md 根層輕薄＋子目錄加層
    -   在子目錄啟 Claude，而非永遠 repo 根
    -   `.claude/settings.json` 的 `permissions.deny` 排除 build artifacts
    -   有 LSP 就開（grep 對常見函式名會爆上下文）
2.  **主動維護 CLAUDE.md（隨模型演進）**
    -   每 3–6 個月回顧一次
    -   曾為舊模型寫的「補丁規則」，新模型可能已不需要甚至有害
3.  **明確歸屬人（DRI）**
    -   沒人管就會部落化；至少需要一位 DRI 管 CLAUDE.md 規範、plugin 市場、權限策略

### 10.4 企業級案例：Anthropic 開源金融 Agent 架構約束
若要將 Agent 部署於高風險的商業環境（如華爾街金融分析），可參考 Anthropic 開源的 `anthropics/financial-services` 框架。其應對真實業務流的核心設計約束如下：
- **寫入權限隔離**：架構分為 System Prompt、Skills 與 Connectors 三層；只有最後一級的 leaf worker（如 deck-writer）持有 Write 權限，上游若被污染也只能影響中間態。
- **數字可追溯性**：每個數字必須能追溯到輸入源；計算儲存格只能是公式而不能有硬編碼，建置完成後必須跑 `audit-xls` 等完整性校驗。
- **資料紀律**：Skills 中明確禁止將 Web Search 作為金融數據的一級來源；必須優先使用 MCP 接出的機構級資料源（如 CapIQ、FactSet、LSEG），不可用時才降級到 EDGAR filings。

---

<a id="11-避坑指南-失敗模式對照表"></a>
## 11. 避坑指南：失敗模式對照表

| 症狀 | 你做錯的事 | 解法 |
| :--- | :--- | :--- |
| Claude 改出一堆你沒要的 | 需求太模糊 | 說清範圍＋限制＋驗證條件 |
| 兩件事的碼混一起、難回滾 | 不 commit 就續做 | 每完成一個階段就 `git commit` |
| Claude 開始「胡說八道」 | 上下文 90%+ 還硬撐 | 70%→`/compact`；90%→`/clear` 重開 |
| 看起來合理但邊界壞掉 | 沒給驗證（測資/截圖/腳本） | 永遠附成功標準 |
| 同一 session 混不相關任務 | 上下文污染 | `/clear`；不相關＝新 session |
| 同一錯法被糾正兩次還錯 | 上下文已被失敗路徑污染 | 先用 `/rewind` 回退到錯之前的節點重下指令；若污染太深再 `/clear` |
| CLAUDE.md 太肥→Claude 忽略一半 | 把該進 Skill 的塞進去了 | 無情刪減；只留普遍且 Claude 猜不到的 |

---

<a id="12-生態擴展一free-claude-code--claude-code-的換引擎本地閘道"></a>
## 12. 生態擴展一：free-claude-code —— Claude Code 的「換引擎」本地閘道

由於官方綁定 Anthropic 且具備嚴格的 Rate Limit，透過部署開源的 **Free Claude Code (FCC)**，你可以將流量代理到全球各大模型，實現模型自由與成本控制。

> **定位**：free-claude-code 專注於「換模型後端」，讓本地 Agent 可以擺脫官方 API 的限制。
> **專案連結**：[https://github.com/Alishahryar1/free-claude-code](https://github.com/Alishahryar1/free-claude-code)

### 12.1 先講清楚它「不是」什麼

它不是：

-   一個「憑空變出無限免費頂級 Claude」的魔法
-   也不需要改 Claude Code 本體

它**是**：

> 一個跑在你本機的 **Anthropic-compatible proxy / gateway**，把 Claude Code 送出的 Messages API 流量轉到你指定的 provider（免費層 / 付費 API / 本地模型）。

> 在本地建立一個 FastAPI 代理伺服器。它攔截 Claude Code 送出的 API 請求，將 OpenAI 的 Chat Streaming 翻譯為 Anthropic SSE 格式，同時正規化 Thinking 區塊與 Tool calls，讓 Claude Code 毫無察覺地使用非官方模型。

* **優勢：** 內建自動壓縮視窗（寫入 `CLAUDE_CODE_AUTO_COMPACT_WINDOW=190000`），並提供網頁版 Admin UI 管理配置。

一句話點破：

> **免費只是鉤子，選擇權才是正文。**  
> 它把「前台（Claude Code 工作流）」和「後台（哪家模型/本地模型）」拆開了。

### 12.2 架構長怎樣（概念圖→文字版）

```
Claude Code CLI / VSCode / Bots
             │
             ▼
  ┌──────────────────────┐
  │  fcc-server (proxy)  │ ← Admin UI (localhost:8082/admin)
  │  - model router      │
  │  - provider adapters │
  │  - request normalize │
  └──────────┬───────────┘
             ▼
  NVIDIA NIM / OpenRouter / Gemini / DeepSeek / Mistral / Kimi
  / Groq / Cerebras / Fireworks / Z.ai / Wafer …
  └─或本地：LM Studio / llama.cpp / Ollama
```

關鍵實作細節：

-   暴露 Anthropic-compatible routes：`/v1/messages`、`/v1/messages/count_tokens`、`/v1/models`
-   支援 streaming / tool use / thinking block 轉譯
-   提供者分兩大類：OpenAI-compat 風格 vs Anthropic Messages 風格 vs 本地路徑

### 12.3 核心設定（最小可跑）

```bash
# 1. 裝官方 Claude Code（它仍需要存在）
npm install -g @anthropic-ai/claude-code

# 2. 裝 free-claude-code
# macOS/Linux
curl -fsSL "https://github.com/Alishahryar1/free-claude-code/blob/main/scripts/install.sh?raw=1" | sh
# Windows PS
irm "https://github.com/Alishahryar1/free-claude-code/blob/main/scripts/install.ps1?raw=1" | iex

# 3. 啟 proxy
fcc-server
# 終端會印 Admin UI URL，預設如 http://127.0.0.1:8082/admin

# 4. 讓 Claude Code 走 proxy（使用官方專屬啟動器）
fcc-claude   
# 註：fcc-claude 會自動讀取你的 Admin UI 設定（包含 Port 與 Token），
# 自動在背景注入環境變數，並幫你把自動壓縮視窗設為 190k tokens，接著啟動真正的 claude。
# (完全不需要手動 export 環境變數！)
```

> **(補充：如果你是在 VS Code 或 JetBrains IDE 內建的 Claude 擴充套件中使用，才需要手動將 `ANTHROPIC_BASE_URL=http://localhost:8082` 與 `ANTHROPIC_AUTH_TOKEN=freecc` 寫入編輯器的 settings.json 中。)**

在 **Admin UI** 裡:

-   選一個 provider（OpenRouter / Gemini / DeepSeek / NVIDIA NIM / local…）
-   貼 API key
-   設 `MODEL=` 對應 slug
-   Validate → Apply

進階：可用 `MODEL_OPUS / MODEL_SONNET / MODEL_HAIKU` 做**分級路由**（重活走強模型、輕活走便宜/本地）。

### 12.4 實用玩法（但請務實期待）

| 玩法 | 價值 | 但你要知道 |
| :--- | :--- | :--- |
| NVIDIA NIM 等免費層 | 日常輕量開發能跑 | 非無限、限速；「免費 Claude Code」嚴格說是「免 Claude 原廠 token 費」，不是無限頂級智力 |
| 本地模型接管 | 隱私好、離線可跑 | 代碼能力/tool calling 穩定性/上下文長度要看你選的模型 |
| 混合路由 | 成本調度 | 工程上很香；但「前台工作流體驗」會被後端模型能力天花板限制 |

### 12.5 風險與安全底線（很重要）

-   這類 proxy 讓 Claude Code 的流量經你本機程式，**你要負責 key 保護、allowed directory、不要亂開 raw logging**
-   `.env` 裡可能有 payload/SSE/CLI diagnostic 開關 → **能不開就不開**，尤其公司 code / 客戶專案
-   bot wrapper（Discord/Telegram）更要設 `allowed directory`、權限邊界
-   **先拿 demo repo 試，不要第一晚就對核心專案開 full 權限**

---

<a id="13-生態擴展二cc-connect--把本地-agent-橋接到你的聊天平台"></a>
## 13. 生態擴展二：cc-connect —— 把本地 Agent 橋接到你的聊天平台

當你需要離開座位，或希望讓整個團隊在通訊軟體中監控並指揮 AI Agent 時，**CC-Connect** 是最強大的橋接中樞。它能將運行在機器上的 Agent（如 Claude Code, Kimi CLI, Gemini CLI 等）對接到 12+ 種通訊平台。

> **定位**：cc-connect 專注於「把 Agent 橋接到聊天平台」（如 Line/飛書/釘釘/Telegram/Discord/微信個人號/QQ 等），負責終端交互的延伸。
> **專案連結**：[https://github.com/chenhg5/cc-connect/](https://github.com/chenhg5/cc-connect/blob/main/README.zh-CN.md)

### 13.1 它解決什麼

一句話：

> 在任何聊天工具裡，遠程操控你的本地 AI Agent；隨時隨地用手機發訊，讓它在你機器上的專案目錄做事。

### 13.2 亮點功能速覽

-   **12 大聊天平台**接入（多數不需公網 IP，靠 WS/long-poll）
-   **10+ Agent 支援**：Claude Code、Codex、Cursor Agent、Gemini CLI、Kimi CLI 等，也支援 ACP
-   **聊天即控制**：斜槓命令切模型 (`/model`)、切權限模式 (`/mode`)、切工作目錄 (`/dir`)
-   **多 Agent 協作**：群裡綁多個機器人，讓 Claude 和 Gemini 在一個對話裡接力
-   **定時任務**：自然語言創建 cron 任務
-   **語音/圖片**：支援 STT/TTS 和多模態轉發
-   **Web 管理後台**：可視化管理項目、服務商、會話、定時任務
-   **持久化記憶**：直接在聊天中讀寫 Agent 的指令文件 (`/memory`)
-   **系統用戶隔離**：用不同 Unix 用戶啟動 Agent，實現作業系統級文件系統隔離
-   **附檔回傳**：Agent 在本地生成圖片/文件後，可主動發回當前聊天

### 13.3 核心玩法：用聊天操控你的本地開發機

**場景**：你在通勤路上，手機收到一個 bug 報告。打開飛書/Telegram，對接了你開發機的 cc-connect bot：

```text
你：/dir ~/projects/myapp
你：檢查一下 src/api/auth.js 的 token 刷新邏輯，用戶報告逾時後登入失敗。
(bot 轉發給本地 Claude Code → Claude Code 讀檔、分析、回覆)
Claude via bot：找到問題。第 45 行的 refreshToken 邏輯沒處理 429 回應。建議修復為… 要現在改嗎？
你：/mode yolo
你：修復它，並跑一遍現有測試。
```

**這就是 cc-connect 的核心價值**：**遠程、非同步、對話式**地驅動你本地的強力編程 Agent，**無需 SSH 連回機器、無需開 IDE**。

### 13.4 安裝與起步

```bash
# 透過 npm（最簡單）
npm install -g cc-connect

# 或 Homebrew
brew install cc-connect

# 啟動 Web 管理後台進行配置（推薦）
cc-connect web
# 瀏覽器會打開管理界面，可視化添加專案、平台、Agent。

# 配置完成後，啟動服務
cc-connect
```

在 Web UI 中，你可以：
1.  **創建專案**：綁定一個本地目錄和一個 AI Agent（如 Claude Code）。
2.  **添加平台**：選擇飛書、Telegram 等，按指南配置 token/webhook。
3.  **管理服務商**：配置 Claude/OpenAI 等 API 金鑰。
4.  **直接對話**：在瀏覽器裡就能和綁定的 Agent 測試對話。

### 13.5 與 free-claude-code 的關係與結合

它們是**互補**而非互斥的：

1.  **free-claude-code** 管**模型後端**：決定 Claude Code 的「大腦」用哪家 API/本地模型。
2.  **cc-connect** 管**訪問入口**：決定你**如何**（透過哪個聊天 App）去觸發和操控 Claude Code。

你可以同時使用：
-   用 `free-claude-code` 讓 Claude Code 對接低成本/本地模型。
-   用 `cc-connect` 把這個「強化版」Claude Code 橋接到你的團隊聊天工具，實現移動辦公。

---

<a id="14-附錄-指令命令速查表"></a>
## 14. 附錄：指令 / 命令速查表

### 14.1 上下文與會話管理

| 命令 | 作用 | 備註 |
| :--- | :--- | :--- |
| `/compact` | 壓縮對話歷史 | 上下文 >70% 時用 |
| `/clear` | 完全重置對話 (保留文件) | 上下文 >90% 或切新任務時用 |
| `/btw` | 快速提問 (不進歷史) | 節省上下文 |
| `Esc` | 停止當前動作 | 保留上下文 |
| `Esc+Esc` 或 `/rewind` | 回退到之前的檢查點 | 後悔藥 |
| `claude --continue` | 在新終端繼續最近會話 | |
| `/rename <名稱>` | 為當前會話命名 | 方便管理 |

### 14.2 專案與文件操作

| 命令/操作 | 作用 |
| :--- | :--- |
| `@filename` | 在對話中引用文件內容 |
| `/init` | 分析專案並生成 CLAUDE.md 初稿 |
| 在**子目錄**啟動 `claude` | 最佳實踐：限定上下文範圍，Claude 會自動向上讀取父級 CLAUDE.md |

### 14.3 核心工作流命令 (部分需 Skills)

| 命令 | 對應 Skill/功能 | 作用 |
| :--- | :--- | :--- |
| `Shift+Tab` (x2) | Plan Mode | 進入「只讀規劃」模式 |
| `/brainstorming <需求>` | Superpowers: Brainstorming | 強制頭腦風暴 |
| `/writing-plans <需求>` | Superpowers: Writing Plans | 生成詳細實施計畫 |
| `/executing-plans` | Superpowers: Executing Plans | 按計畫逐步執行 |
| `/test-driven-development <需求>` | Superpowers: TDD | 以測試驅動的方式開發 |
| `/systematic-debugging <現象>` | Superpowers: Systematic Debugging | 結構化四階段除錯 |
| `/requesting-code-review` | Superpowers: Code Review | 5專項Agent並行審查 |
| `/dispatching-parallel-agents <任務>` | Superpowers: Parallel Agents | 並行處理獨立子任務 |
| `/verification-before-completion` | Superpowers: Verification | 完工前強制檢查清單 |
| `/using-git-worktrees <功能>` | Superpowers: Git Worktrees | 創建隔離的分支開發環境 |
| `/excalidraw-diagram <描述>` | Excalidraw Diagram | 生成可編輯架構圖 |
| `/batch <批量任務>` | (內建) Batch | 批量並行處理任務 |

### 14.4 cc-connect 常用斜槓命令 (在聊天平台中使用)

| 命令 | 作用 |
| :--- | :--- |
| `/new [名稱]` | 創建新會話 |
| `/list` | 列出所有會話 |
| `/switch <id>` | 切換會話 |
| `/model` | 列出可用模型 |
| `/model switch <alias>` | 切換模型 |
| `/mode` | 查看權限模式 |
| `/mode yolo` | 切換到自動批准模式 |
| `/dir` 或 `/cd` | 查看/切換工作目錄 |
| `/dir <路徑>` | 切換到指定目錄 |
| `/cron add <表達式> <任務>` | 添加定時任務 |
| `/provider list` | 列出API服務商 |
| `/provider switch <名稱>` | 切換服務商 |

### 14.5 free-claude-code 關鍵命令

| 命令 | 作用 |
| :--- | :--- |
| `fcc-server` | 啟動本地代理伺服器和 Admin UI |
| `fcc-claude` | 啟動已配置代理的 Claude Code 客戶端 |
| 訪問 `http://127.0.0.1:8082/admin` | 打開管理界面配置模型和服務商 |

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "2026 Claude Code 完全整合指南：從核心機制、工程紀律、大規模部署到生態擴展",
  "description": "深度整合了多篇 Claude Code 核心文章與開源項目，提供一套可查閱、可執行、有層次的全景知識體系。",
  "author": {
    "@type": "Person",
    "name": "TonTon Huang Ph.D.",
    "url": "https://twman.org"
  },
  "keywords": ["Claude Code", "AI Agent", "CLI", "Developer Tools", "Anthropic"],
  "inLanguage": "zh-Hant",
  "articleSection": "Software Engineering",
  "proficiencyLevel": "Expert"
}
</script>