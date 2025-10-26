---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---


{% include header.html %}

---

{% include ai-share.html %}

---

# AIxCC 冠軍 Atlanta 團隊「ATLANTIS」系統關鍵重點解析

> 同步匯整自 [https://zread.ai/Team-Atlanta/aixcc-afc-atlantis](https://zread.ai/Team-Atlanta/aixcc-afc-atlantis)
> 同步匯整自 [`Team-Atlanta/aixcc-afc-atlantis`](https://github.com/Team-Atlanta/aixcc-afc-atlantis)：包含 AIxCC 總決賽中提交的原始程式碼
> 同步匯整自 [AIxCC狀元探秘](https://mp.weixin.qq.com/s/Ffy1e6ojxj1VhrigQvtgHw)

針對 DARPA AI 網路挑戰賽（AIxCC）解析冠軍團隊 Team-Atlanta 所開發的網路推理系統（CRS）——「ATLANTIS」的關鍵技術、架構和核心策略。

***

## 目錄

1.  [團隊與系統簡介](#1-團隊與系統簡介)
2.  [ATLANTIS 核心理念與策略](#2-atlantis-核心理念與策略)
3.  [系統高層架構](#3-系統高層架構)
4.  [核心解析：自動化補丁 Agent 策略 (Ensemble of Agents)](#4-核心解析-自動化補丁-agent-策略-ensemble-of-agents)
    * [4.1 MartianAgent (工作流驅動)](#41-martianagent-工作流驅動)
    * [4.2 MultiRetrievalPatchAgent (多策略檢索)](#42-multiretrievalpatchagent-多策略檢索)
    * [4.3 PrismAgent (多團隊協作狀態機)](#43-prismagent-多團隊協作狀態機)
    * [4.4 VincentAgent (Sanitizer/PoC 驅動)](#44-vincentagent-sanitizerpoc-驅動)
    * [4.5 ClaudeLikeAgent (ReAct 反饋迴圈)](#45-claudelikeagent-react-反饋迴圈)
    * [4.6 EraserAgent (策略驅動與環境模擬)](#46-eraseragent-策略驅動與環境模擬)
5.  [漏洞挖掘：Fuzzing 優化策略](#5-漏洞挖掘-fuzzing-優化策略)
    * [5.1 UniAFL (多語言 Fuzzing 框架)](#51-uniafl-多語言-fuzzing-框架)
    * [5.2 Directed Jazzer (導向 Fuzzing)](#52-directed-jazzer-導向-fuzzing)
6.  [總結：關鍵啟示](#6-總結-關鍵啟示)

***

## 1. 團隊與系統簡介

* **團隊名稱：** Team Atlanta
* **團隊負責人：** Taesoo Kim
* **團隊背景：** 由喬治亞理工學院（Georgia Tech）、三星研究院（Samsung Research）、KAIST 和 POSTECH 的專家組成。
* **獲獎成就：** 在 DEF CON 33 (2025 年 8 月) 舉辦的 DARPA AI 網路挑戰賽（AIxCC）總決賽中榮獲第一名。
* **參賽系統：** ATLANTIS (AI-driven Threat Localization, Analysis, and Triage Intelligence System)。

## 2. ATLANTIS 核心理念與策略

ATLANTIS 的成功並非依賴單一技術，而是多種策略的「集成（Ensemble）」。

1.  **正交策略（Orthogonal Strategies）：** ATLANTIS 並非單一的 CRS，而是由多個獨立的 CRS（如 C、Java、Multilang、Patch）組成。團隊刻意追求不同的（正交的）策略，以最大化覆蓋範圍和容錯能力。
2.  **LLM 作為「增益品」：** 團隊的結論是，大型語言模型（LLM）是傳統程式分析方法（如 Fuzzing、符號執行、靜態分析）的「增益品」，而非「替代品」。兩者必須緊密結合。
3.  **模型大小的「甜蜜點」：** 研究發現，並非模型越大越好。像 GPT-4o-mini 這樣的 8B 參數模型 達到了一個「甜蜜點」：它足夠大，能理解程式碼模式、API 約定和異常堆疊；同時也足夠小，輸出更簡潔，決策更直接，不易將簡單問題複雜化。
4.  **多樣性集成（Ensembling）：** 系統在各個層面都採用了集成策略，包括 Fuzzing、符號執行和補丁生成。特別是在補丁生成方面，團隊利用了 LLM 的「非確定性」（或稱「幻覺」），將其視為一種「創造力」，透過集成多種 Agent 來駕馭這種特性。

## 3. 系統高層架構

根據 CRS 架構圖（Design Overview of Atlantis），系統基於 K8s 節點 運行，主要組件包括：

* **CRS Web Server / CRS DB：** 接收來自 `AIXCC API` 的挑戰（Challenges）。
* **Log Collector：** 收集 `OpenTelemetry Logs`。
* **LLM Proxy：** 作為 LLM 請求的中介，管理 `LiteLLM` 和 `Custom LLM`。
* **Bug Finding Modules：** 核心的漏洞發現單元，包含 `ATLANTIS-C`、`ATLANTIS-JAVA` 和 `ATLANTIS-MULTILANG`。
* **CP Manager：** 挑戰包（Challenge Package）管理器，負責建構、預算分配、任務管理、PoV 驗證等。
* **Atlantis-Patch：** 接收 CP Manager 的任務，專門負責生成補丁（Patch）。
* **Atlantis-SARIF：** 負責處理 SARIF 評估（Assessment）並提交報告。

## 4. 核心解析：自動化補丁 Agent 策略 (Ensemble of Agents)

`ATLANTIS-PATCHING` 模塊 如何集成多種具有不同策略的 Agent 來生成補丁。這體現了其「集成」和「正交策略」的核心思想。

### 4.1 MartianAgent (工作流驅動)

* **核心策略：** 根據「預設工作流」自動定位、分析並修復安全缺陷。
* **執行流程：** `Crash Log` -> `Fault Localization` (故障定位) -> `Bug Analysis Report` (錯誤分析報告) -> `Patch Generation` (補丁生成)。
* **LLM 使用：** 為不同任務指派了不同的 LLM：
    * **漏洞定位：** `o4-mini`。
    * **報告解析：** `gpt-4o`。
    * **程式碼生成：** `claude-sonnet-4-20250514`。
    * **備用模型：** `gemini-2.5-pro`。

### 4.2 MultiRetrievalPatchAgent (多策略檢索)

* **核心策略：** 以「多策略程式碼檢索 + 補丁生成」為核心。
* **執行流程：** 強調在生成補丁 *之前* 必須先進行分析、探索和檢索。它採用多種檢索手段（如正則、AST、語義等），並在 Prompt 中明確要求每輪檢索 3-5 條資訊，優先使用 `grep`，輔以 `file` 查詢。
* **適用場景：** 適合需要複雜資訊檢索、上下文理解和多輪推理的場景。

### 4.3 PrismAgent (多團隊協作狀態機)

* **核心策略：** 採用「多團隊協作 + 狀態機工作流」，實現自動化的分析、補丁、評測的多輪閉環。
* **架構：** 模擬一個組織架構，包含四個「團隊」（節點）：
    * `Supervisor` (主管)：作為路由節點，根據當前狀態（如評測次數、補丁狀態）動態路由到下一個團隊。
    * `Analysis Team` (分析團隊)：負責漏洞分析。
    * `Patch Team` (補丁團隊)：負責生成補丁。其內部還有 `patch_generator`（生成）和 `patch_reviewer`（評審）的子迴圈。
    * `Evaluation Team` (評估團隊)：負責補丁評測。

### 4.4 VincentAgent (Sanitizer/PoC 驅動)

* **核心策略：** 基於 `sanitizer` 報告和 `PoC`（漏洞利用範例）資訊來自動分析和修復。
* **執行流程：** 使用 Langchain 框架實現，其工作流（Workflow）是一個狀態圖，包含 `Root Cause Analysis`（根本原因分析）、`Property Analysis`（屬性分析）、`Patch Generation`（補丁生成） 以及多種反饋節點（如 `compile_feedback`、`vulnerable_feedback`），實現「分析-生成-反饋-再生產」的閉環。
* **工具：** 明確使用了 `tree-sitter` 作為程式碼解析工具。

### 4.5 ClaudeLikeAgent (ReAct 反饋迴圈)

* **核心策略：** 以 `ReAct`（結合推理與行動）智能體為核心，並整合多工具和多輪反饋。
* **執行流程：** 最大的特色是其反饋機制。系統會自動評測生成的補丁 `diff`。如果評測結果為失敗類型（如 `VulnerableDiffAction`、`UncompilableDiffAction` 等），它會將本輪失敗的 `diff` 內容追加到 `failed_patches` 變數中，並將其作為*輸入*反饋給下一輪的 Prompt，從而實現多輪迭代優化。
* **LLM 使用：** 使用 `claude-3-7-sonnet-20250219` 模型。

### 4.6 EraserAgent (策略驅動與環境模擬)

* **核心策略：** 以「自動化、策略驅動和 LLM 結合」為核心。
* **執行流程：** 它首先構建一個 `AIxCCEnvironment`（自動化修復環境），然後透過 `EraserPolicy`（策略）來驅動環境進行多步交互（`environment.step`）。Policy 決定每一步的 `action`，環境執行後返回新的觀察值（`observation`），直到交互完成。
* **LLM 使用：** 使用 `gpt-4.1` 作為補丁生成模型，並使用一個名為 `models/betarixm/eraser` 的模型作為檢索的後備模型。

## 5. 漏洞挖掘：Fuzzing 優化策略

PDF 的後半部分和總結強調，團隊在傳統程序分析（特別是 Fuzzing）方面進行了大量改進。

### 5.1 UniAFL (多語言 Fuzzing 框架)

* **亮點：** 這是一個支持多語言的 Fuzzing 工具，在決賽中貢獻了 69.2% 的 PoV。
* **LLM 應用：** UniAFL 的架構包含了 6 個輸入生成模塊，展示了 LLM 在 Fuzzing 中的多種應用層次：
    * **不使用 LLM：** 傳統的 `libFuzzer` 和 `Jazzer`，或混合 Fuzzing。
    * **有限使用 LLM：** 應用於字典生成（Dictionary-based） 或將輸入表示為 TestLang 並進行變異（Testlang-based）。
    * **大量使用 LLM：** 利用 LLM 構造污點傳播鏈、識別潛在漏洞並生成輸入（MLLA）。

### 5.2 Directed Jazzer (導向 Fuzzing)

* 展示了一個以 `Sinkpoint`（漏洞觸發點）為核心的導向 Fuzzing 架構，它結合了靜態分析（`CodeQL CG`、`Joern CG`） 和 `Sink Point DB` 來計算距離，並指導 Fuzzer（`Directed Jazzer`） 優先探索可能觸發漏洞的路徑。

## 6. 總結：關鍵啟示

1.  **LLM 必須與傳統工具結合：** LLM 需要與傳統的程式分析工具（Fuzzing、符號執行、靜態分析）深度結合，才能發揮最大效用。
2.  **LLM 是「增益品」，非「替代品」：** LLM 在理解程式碼、API、異常堆疊等方面表現出色，但它們是傳統分析方法的增強，而非取代。
3.  **策略平衡（輕重LLM）：** 應根據目標選擇合適的策略。例如，`Atlantis-C` 採用了「重插樁 / 輕 LLM」的策略，而 `Atlantis-Multilang` 則走了「輕插樁 / 重 LLM」的路線。
4.  **模型大小的「甜蜜點」：** 8B 參數模型被證明在效能和效率上達到了最佳平衡點。
5.  **LLM 在 Fuzzing 中的應用：** LLM 在 Fuzzing 領域的應用已相當具體，包括字典生成、種子生成和輸入生成。