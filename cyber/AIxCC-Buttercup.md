---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

{% include header.html %}

---

{% include ai-share.html %}

---

# 漏洞自動分析與修復Agent:Buttercup 如何打通檢索—分析—修補—驗證

> 同步匯整自 [https://zread.ai/trailofbits/afc-buttercup](https://zread.ai/trailofbits/afc-buttercup)
> 同步匯整自 [`trailofbits/afc-buttercup`](https://github.com/trailofbits/afc-buttercup)：包含 AIxCC 總決賽中提交的原始程式碼
> 同步匯整自 [漏洞自動分析與修復Agent:Buttercup 如何打通檢索—分析—修補—驗證閉環](https://mp.weixin.qq.com/s/3AB2fRQd7QGjMaWXeJABuQ)

`afc-buttercup` 是 Trail of Bits 團隊為 AIxCC 總決賽（AFC）提交的開源版本。該系統將 AI 增強模糊測試、靜態分析（如 tree-sitter/CodeQuery）與一個創新的「多智能體補丁生成—驗證流水線」相整合。最終，`afc-buttercup` 斬獲 AIxCC 總決賽亞軍，並在 20 類 CWE 上實現了高準確度的漏洞發現與修復表現。

***

## 目錄

1.  [Buttercup 核心：多智能體協作架構](#1-buttercup-核心多智能體協作架構)
2.  [Agent 職責與工作流詳解](#2-agent-職責與工作流詳解)
      * [2.1 PatcherLeaderAgent (總控指揮官)](#21-patcherleaderagent-總控指揮官)
      * [2.2 InputProcessingAgent (輸入預處理)](#22-inputprocessingagent-輸入預處理)
      * [2.3 ContextRetrieverAgent (智能上下文補全器)](#23-contextretrieveragent-智能上下文補全器)
      * [2.4 RootCauseAgent (漏洞根因分析)](#24-rootcauseagent-漏洞根因分析)
      * [2.5 SWEAgent (AI 開發工程師)](#25-sweagent-ai-開發工程師)
      * [2.6 QEAgent (品質保障工程師)](#26-qeagent-品質保障工程師)
      * [2.7 ReflectionAgent (自適應反思與補救)](#27-reflectionagent-自適應反思與補救)
3.  [關鍵工作流程：自動修復的閉環實現](#3-關鍵工作流程自動修復的閉環實現)
4.  [核心特性與總結](#4-核心特性與總結)

***

## 1\. Buttercup 核心：多智能體協作架構

Buttercup 的自動化補丁流程並非單一模型，而是由多個各司其職的專職 Agent 協同運作，形成一條高內聚、低耦合的流水線。系統將複雜的補丁任務拆分為上下文檢索、根因分析、補丁生成、測試驗證和失敗反思等獨立環節。

## 2\. Agent 職責與工作流詳解

系統的核心由七個主要的 Agent 組成，由 `PatcherLeaderAgent` 統一編排調度。

### 2.1 PatcherLeaderAgent (總控指揮官)

  * **職責：** 作為補丁流水線的「總控/編排」Agent。
  * **作法：** 負責組裝和調度所有子 Agent。它透過 `_init_patch_team` 方法構建一個多節點的狀態機（StateGraph），定義完整的自動補丁工作流，並啟動和管理整個任務。

### 2.2 InputProcessingAgent (輸入預處理)

  * **職責：** 在補丁生成前，負責輸入環境和上下文的初始化與準備。
  * **作法：** 解析並應用工作目錄等配置，並初始化 `codequery` 等代碼查詢工具，確保後續分析能獲取準確的代碼結構資訊。

### 2.3 ContextRetrieverAgent (智能上下文補全器)

  * **職責：** 作為流水線中的「智能上下文補全器」，動態檢索並補全上下文資訊。
  * **作法：** 當主流程 Agent 發現資訊不足或 LLM 請求更多上下文時，此 Agent 會被自動調用。
      * `get_initial_context`：根據崩潰堆棧資訊，自動提取相關的關鍵代碼片段（函數名、文件名、行號）。
      * `find_tests_node`：透過查找腳本、從 Redis 查找指令或 `find_tests_agent` 自動探索等多種方法，為補丁驗證階段準備可用的測試指令。

### 2.4 RootCauseAgent (漏洞根因分析)

  * **職責：** 負責漏洞根因分析，定位問題本質。
  * **作法：**
      * **LLM 驅動：** 入口為 `analyze_vulnerability`，主要使用 `openai-gpt-4.1` 和 `claude-3.7-sonnet` 模型分析漏洞成因，並解析返回的結構化資訊（如 `root_cause`、`vuln_line` 等）。
      * **工具注入 (MCP)：** 如果 LLM 分析時需要更多上下文，可透過工具注入機制，自動調用一系列工具（如 `ls`、`grep`、`get_function`、`get_type`、`get_callers` 等）來補充資訊。

### 2.5 SWEAgent (AI 開發工程師)

  * **職責：** 作為「AI 開發工程師」，專注於補丁策略制定和代碼修復生成。
  * **作法：**
    1.  **選擇補丁策略 (select\_patch\_strategy)：**
          * Agent 會組裝包含根因分析、代碼片段等的 Prompt，調用 LLM 生成結構化的補丁策略（Patch Strategy）。
          * 如果 LLM 輸出 `<request_information>`，代表需補充上下文，流程會先跳轉至 `ContextRetrieverAgent`。
          * Foucs, 提取 `<full_description>`（詳細設計說明）和 `<summary>`（核心思路概括）作為補丁策略。
    2.  **生成具體補丁 (create\_patch\_node)：**
          * 組裝包含「補丁策略」和根因分析結果的 Prompt，調用 LLM 生成最終的修復補丁。

### 2.6 QEAgent (品質保障工程師)

  * **職責：** 作為「品質保障工程師」，負責補丁生成後的自動化測試與有效性校驗。
  * **作法：**
      * `build_patch_node`： 核心驗證節點。它會調用 `_build_with_sanitizer`，在乾淨環境下應用補丁並「並行」編譯項目（針對每個 sanitizer）。
      * **流程跳轉：** 如果所有 sanitizer 均構建成功，流程進入 `RUN_POV`（運行測試/驗證）階段；如果構建失敗，則進入 `REFLECTION`（反思）階段。

### 2.7 ReflectionAgent (自適應反思與補救)

  * **職責：** 負責反思與補救，是實現流程閉環和魯棒性的關鍵。當主流程遇到異常或失敗（如構建失敗、測試失敗）時，此 Agent 會接管流程。
  * **作法：**
      * `_analyze_failure`： 收集詳細失敗數據（如 `stdout/stderr`），調用 LLM 反思鏈（`self.reflection_chain`）分析失敗原因、歸納失敗類別。
      * **動態調度：** 根據 LLM 的反思建議，自動決定「下一個環節」，例如是退回 `CONTEXT_RETRIEVER` 補充上下文、退回 `ROOT_CAUSE_ANALYSIS` 重新分析根因，還是退回 `SWEAgent` 調整 `PATCH_STRATEGY`。
      * `reflect_on_patch`： 如果 LLM 明確請求更多代碼片段（`code_snippet_requests`），會自動切換到 `CONTEXT_RETRIEVER` 節點補充資訊，反思後再回到原節點繼續。

## 3\. 關鍵工作流程：自動修復的閉環實現

Buttercup 的 Agent 協作形成了一個完整的「檢索-分析-修補-驗證-反思」閉環：

1.  **啟動與預處理 (InputProcessing)：** 初始化環境，準備 `codequery` 工具。
2.  **上下文檢索 (ContextRetriever)：** Berdasarkan `get_initial_context` 堆棧崩潰，獲取初始代碼上下文。
3.  **根因分析 (RootCause)：** `RootCauseAgent` 介入，使用 LLM 和工具集分析漏洞成因。
4.  **補丁生成 (SWEAgent)：** `SWEAgent` 制定 `patch_strategy`（包含 `<full_description>` 和 `<summary>`），然後 `create_patch_node` 生成補丁。
5.  **構建與驗證 (QEAgent)：** `QEAgent` 嘗試並行 `build_patch_node`。
6.  **反思與閉環 (Reflection)：**
      * **若驗證成功：** 流程結束。
      * **若驗D thất bại：** `ReflectionAgent` 接管。它調用 LLM 分析失敗日誌 (`_analyze_failure`)，並做出決策：是 `ContextRetrieverAgent` 補充上下文，還是 `RootCauseAgent` 重新分析，或是 `SWEAgent` 重新制定策略。

## 4\. 核心特性與總結

`afc-buttercup` 是一個以多智能體和大模型為核心，兼具工程實用性與 AI 智能化的自動化安全補丁平台。

  * **多智能體協作架構：** 將流程拆分為多個專職 Agent，實現高內聚、低耦合的流水線。
  * **強大的 LLM 驅動與工具鏈注入：** 透過 LLM（如 GPT-4、Claude）驅動決策，並支持自動調用代碼檢索、`grep`、`get_function` 等工具。
  * **自適應反思與流程調度：** `ReflectionAgent` 能自動分析失敗原因、檢測死循環，並動態調整流程，保證了系統的魯棒性和持續進步能力。
  * **並行與容錯機制：** 支持多 sanitizer 並行 build/test，主備 LLM 自動切換，以及 Agent 層面的重試保護，提升了效率與穩定性。
  * **安全導向與結構化數據流：** 所有決策均聚焦於安全修復，且補丁策略、失敗分析等均採用結構化管理，便於追蹤和優化。