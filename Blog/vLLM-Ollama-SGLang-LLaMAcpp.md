---
layout: default
title: "vLLM, Ollama, SGLang, LLaMA.cpp 深度評測 | LLM 推論框架選型指南"
description: "深度解析四大主流 LLM 推論框架。vLLM 適合高吞吐生產環境，Ollama 適合本地開發，SGLang 專攻複雜 Agent 工作流。"
permalink: /vLLM-Ollama-SGLang-LLaMAcpp
lang: zh-Hant
keywords: ["vLLM", "Ollama", "SGLang", "LLM 推論", "GPU 加速"]
---


{% include header.html %}

---

{% include ai-share.html %}

---

**作者**：[TonTon Huang Ph.D.](https://twman.org)  

---

# vLLM、Ollama、SGLang、 LLaMA.cpp
_深度解析四大主流熱門LLM服務框架_

> **🚀 本文重點摘要 (TL;DR)：**
> 如何選擇 LLM 推論框架？
> * **vLLM**：生產環境首選，高吞吐量與低延遲。
> * **Ollama**：開發者首選，安裝最簡單，支援 Mac。
> * **SGLang**：Agent 應用首選，優化複雜結構化輸出。
> * **LLaMA.cpp**：邊緣設備首選，支援純 CPU 與低資源環境。

生產環境高吞吐與低延遲選 vLLM；本地快速上手與多模型管理選 Ollama；複雜代理/結構化工作流選 SGLang；極致輕量與可攜性選 LLaMA.cpp Server。以下從架構原理、性能優化、特性矩陣、部署與運維到選型決策提供完整分析。

### 四大框架特性矩陣

| 維度 | Ollama | vLLM | SGLang | LLaMA.cpp Server |
|---|---|---|---|---|
| 核心定位 | 本地便捷與模型管理 | GPU 高吞吐 / 低延遲 | 複雜工作流 + 高性能 | 極致輕量、隨處可跑 |
| 典型硬體 | CPU / Apple Silicon / NVIDIA | NVIDIA CUDA 多卡 / 多機 | NVIDIA / 部分 AMD GPU | CPU / CUDA / Apple Metal / WASM |
| 權重 / 格式 | GGUF、亦可導入 HF 權重 | HF Transformers / safetensors | HF Transformers / safetensors | GGUF（主） |
| 量化 | 4 / 5 / 8-bit（GGUF） | FP16 / BF16（外掛支援 GPTQ / AWQ / FP8） | FP16 / BF16 / INT4 / FP8 / GPTQ | 4 / 5 / 8-bit（GGUF） |
| KV Cache 優化 | 基於 llama.cpp，支援 paged KV cache 與流式管理 | PagedAttention | RadixAttention + Reuse Cache（支援 streaming prefill） | 高效 C++ 實作 |
| 批次 / 排程 | 基礎，單模型單隊列偏多 | 連續批次 + 動態排程 | 連續批次 + 零開銷排程 | 單隊列為主 |
| 多 GPU / 分散式 | 有限 | 強 | 強（張量並行） | 有限（以單機為主） |
| 多模型 / 多租戶 | 易切換，併發有限 | 支援，多模型常駐 / 熱切換 | 支援，工作流級控制 | 支援多模型切換（非並行） |
| LoRA / Adapter | 基本支援 | 多 LoRA / PEFT 請求級 | LoRA / Adapter 支援 | 具 LoRA 推理支援 |
| 長上下文 | 取決於模型與量化，速度中等 | 高效，適合長上下文高併發 | 高效，Chunked Prefill 佳 | 可長上下文，吞吐較低 |
| 推測解碼 | 限 | 有（逐步完善） | 有（成熟） | 有 |
| 結構化 / 約束輸出 | 基礎 | JSON / 函式工具模式 | JSON、函式與自定義 DSL（最強） | Grammar / GBNF 成熟 |
| 多模態 | 取決於模型包 | 支援多模態模型（視模型） | 支援文字 / 多模態管線 | 取決於模型與轉換 |
| OpenAI API 兼容 | 是 | 是 | 是 | 是 |
| 嵌入 / 向量 | 有 | 有 | 有 | 有 |
| 監控 / 可觀測 | 基礎 | 較完善（指標 / 日誌） | 較完善（工作流視角） | 基礎 |
| 部署複雜度 | 極低 | 中（需 GPU 與調優） | 中（需 GPU + 程式化） | 低（單一二進位） |
| 社群成熟度 | 高 | 高 | 中高（增長快） | **極高** |
| 代表用例 | 私有助手 / 離線 / PoC | 生產級 API 服務 | 代理 / 工具協作 / 多步任務 | 邊緣 / 離線 / 受限環境 |

## 核心技術機制對比

| 維度 | vLLM | SGLang | Ollama / LLaMA.cpp | TensorRT-LLM |
|---|---|---|---|---|
| **記憶體/快取** | **PagedAttention** (分頁虛擬記憶體) | **RadixAttention** (字首樹共享) | **量化** (GGUF, 權重壓縮) | 核心級優化 |
| **批次/排程** | **連續批次** (動態插入) | **連續批次** + 零開銷排程 | 單隊列為主 (Ollama) | 優化的批次處理 |
| **量化支援** | FP16/BF16 (外掛 GPTQ/AWQ) | FP16/BF16/INT4/FP8 | **GGUF** (4/5/8-bit) | **FP8 / FP4 / INT4** (原生) |
| **結構化輸出** | JSON / 函式工具模式 | **DSL 驅動** (最強) | **GBNF** (LLaMA.cpp) / 基礎 (Ollama) | 支援有限 |

## 框架選型總覽表

| 框架 | 核心技術/優勢 | 典型適用場景 |
|---|---|---|
| **vLLM** | PagedAttention, 連續批次, TTFT優異 | 企業級高併發, 生產級 API 服務 |
| **SGLang** | RadixAttention (前綴複用), 結構化 DSL | 複雜工作流, 代理/多步驟任務, 高吞吐多輪對話 |
| **Ollama** | 易用, 本地部署, 多模型管理 (GGUF) | 個人開發, 快速原型, 隱私/離線場景 (Apple Silicon/CPU) |
| **LLaMA.cpp Server** | C++ 實現, 極致輕量, GBNF 語法約束 | 邊緣設備, 硬體受限環境, 跨平台 (WASM) |
| **TensorRT-LLM** | NVIDIA 深度優化, 強大量化 (FP8/FP4), 延遲最低 | 對延遲要求極苛刻的應用 (如高頻交易) |
| **XInference** | 分離式部署 (Prefill/Decode), K8s 分布式 | 大規模分布式部署, 快速驗證 |
| **LightLLM** | 三進程異步, TokenAttention, 輕量級 | 邊緣設備部署 (手機, IoT) |
| **LMDeploy** | 國產硬體 (昇騰) 深度優化, 多模態 | 國產硬體部署, 視覺語言混合任務 |
| **MindSpore Inference** | 昇騰達芬奇架構, CBQ 量化 | 昇騰硬體生態 |

# 主流大模型推理部署框架全面梳理
本文系統性地梳理了當前主流的大模型推理部署框架，深度解析 vLLM、Ollama、SGLang、LLaMA.cpp 及 TensorRT-LLM 等框架的核心技術、架構設計、性能優化與適用場景，並提供完整的選型決策分析。

## 核心框架深度解析

以下我們將深入探討幾個最受關注的框架，並補充其他重要的專業框架。

### 1. vLLM：基於 PyTorch 的高性能推理引擎
vLLM 專為 GPU 伺服器上的高吞吐 LLM 推理而設計，是企業級部署的首選之一。

* **核心技術**：
    * **PagedAttention（分頁注意力）**：借鑒作業系統的分頁機制，將 KV Cache 儲存在非連續的顯存空間（頁式虛擬記憶體）。這有效解決了顯存碎片問題，將顯存利用率從 60% 提升至 95% 以上，顯著減少了因記憶體過度配置導致的浪費。
    * **Continuous Batching（連續批處理）**：允許在批次處理過程中動態插入新的請求，確保 GPU 保持持續忙碌狀態，大幅提升吞吐量。
* **其他特性**：支援多 GPU 擴展、LoRA 多適配器、以及 OpenAI 風格的 JSON 模式與函式（Tool）呼叫。
* **適用場景**：企業級高併發應用，如線上客服、生產級 API 服務等對延遲與吞吐量要求極高的場景。

### 2. SGLang：面向複雜工作流的程式化引擎
SGLang (Structured Generation Language) 由 LMSYS 團隊開發，定位為面向複雜、多步驟、可結構化的 LLM 程式化工作流引擎。

* **核心技術**：
    * **RadixAttention（基數注意力）**：利用 Radix 樹（字首樹）來管理和共享 KV 快取的前綴。這使得在多分支、多步驟的代理（Agent）流程中，能高效地跨請求複用快取，顯著提升複雜任務的吞吐量（在多輪對話場景下可達 vLLM 的數倍）。
    * **結構化輸出 (DSL)**：提供前端 DSL（領域特定語言），可強力約束模型生成 JSON、函式呼叫或自定義格式，在多步驟協調上表現最強。
* **其他特性**：支援推測解碼、張量並行、零開銷排程等。
* **適用場景**：需要高吞吐量的複雜工作流，如代理（Agent）應用、工具協作、多步驟任務、或需要嚴格結構化輸出的場景。

### 3. Ollama：輕量級本地推理與管理平台
Ollama 注重本地部署的易用性與跨平台體驗，是個人開發者與快速原型的首選。

* **核心技術**：
    * **Go 語言封裝**：底層整合 llama.cpp/ggml/gguf 生態，並以 Go 語言封裝，提供一鍵部署的流暢體驗（冷啟動僅需 12 秒左右）。
    * **多模型管理**：支援 `Modelfile` 來自定義模型、系統提示與參數，便於管理和切換本地的多個模型。
* **其他特性**：支援 CPU、Apple Silicon (Metal GPU) 及 NVIDIA CUDA。支援完全離線運行，確保數據安全與隱私。
* **適用場景**：個人開發者、教育展示、本地隱私要求高、或在 Apple Silicon 上運行的場景。

### 4. LLaMA.cpp Server：極致輕量的本地伺服器
LLaMA.cpp 是以純 C/C++ 實現的高效推理實作，其 `server` 模式提供了極致輕量級的部署方案。

* **核心技術**：
    * **純 C/C++ 實作**：依賴極低，可編譯為單一二進位檔案，具備極高的可攜性。
    * **GGUF 格式與量化**：深度支援 4/5/8-bit 的 GGUF 量化格式，極大降低記憶體占用。
    * **GBNF 語法約束**：支援 `Grammar/GBNF` 約束，可嚴格控制模型輸出格式，在邊緣端生成結構化資料時非常實用。
* **其他特性**：支援 CPU、CUDA、Apple Metal，甚至 WASM (WebAssembly)。
* **適用場景**：硬體資源極受限的環境、邊緣設備、需要極致可攜性或離線運行的應用。

### 5. TensorRT-LLM：NVIDIA 深度優化推理引擎
這是 NVIDIA 官方推出的深度優化框架，專注於挖掘 NVIDIA GPU 的極致性能。

* **核心技術**：
    * **預編譯與核心級優化**：通過 TensorRT 進行全鏈路優化，生成高度優化的引擎檔案，延遲表現通常是最佳的。
    * **強大量化支援**：支援 FP8、FP4 和 INT4 等多種低精度量化方案，顯存占用可減少 40% 以上。
* **適用場景**：對響應延遲要求極度苛刻的企業級應用，如即時客服系統、金融高頻交易等。

### 6. XInference：分布式推理框架
XInference 專為企業級大規模部署設計，特別強調其分布式能力。

* **核心技術**：
    * **分離式部署**：架構上支援將 Prefill（提示處理）和 Decode（生成）階段分配到不同的 GPU 上運行，優化資源利用。
    * **K8s 擴展**：支援 Kubernetes 集群擴展，並結合 vLLM 的連續批處理技術優化請求調度。
* **適用場景**：企業級大規模部署、智能客服系統、知識庫問答，或需要快速驗證的分布式場景。

### 7. LightLLM：輕量級高性能框架
此框架專為輕量化和邊緣部署設計。

* **核心技術**：
    * **三進程異步協作**：獨特的架構設計，平衡吞吐量和延遲。
    * **TokenAttention**：針對 KV Cache 的優化機制。
* **適用場景**：邊緣設備部署，如智能手機和 IoT 設備。

## 總結與選型建議

大模型推理部署框架的選擇應基於 **業務需求、硬體資源和未來擴展規劃** 綜合考慮：

1.  **企業級高併發與低延遲 (NVIDIA GPU)**：
    * **vLLM** 是高吞吐 API 服務的首選。
    * **TensorRT-LLM** 適用於對 P99 延遲要求最為苛刻的場景。

2.  **複雜工作流與高吞吐 (NVIDIA GPU)**：
    * **SGLang** 在代理（Agent）、工具編排或需要嚴格結構化輸出的多步驟任務上具有明顯優勢。

3.  **個人開發/本地/隱私優先**：
    * **Ollama** 提供最佳的易用性、模型管理和跨平台（尤其是 Apple Silicon）體驗。
    * **LLaMA.cpp Server** 適用於需要極致輕量、低依賴或 GBNF 語法約束的本地場景。

4.  **邊緣/硬體受限/跨平台**：
    * **LLaMA.cpp Server** 憑藉其 C++ 核心和 GGUF 格式，是資源受限環境的首選。
    * **LightLLM** 專為手機、IoT 等邊緣設備設計。


<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/vLLM-Ollama-SGLang-LLaMAcpp"
  },
  "headline": "vLLM、Ollama、SGLang、LLaMA.cpp：深度解析四大主流熱門LLM服務框架",
  "description": "深度解析並比較 vLLM, Ollama, SGLang, LLaMA.cpp 等四種主流大型語言模型（LLM）服務框架，內容涵蓋架構原理、性能優化、特性比較、部署運維及選型決策指南。",
  "image": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg",
  "author": {
    "@type": "Person",
    "name": "TonTon Huang Ph.D.",
    "url": "https://twman.org"
  },
  "publisher": {
    "@type": "Organization",
    "name": "Deep Learning 101, Taiwan",
    "logo": {
      "@type": "ImageObject",
      "url": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg"
    }
  },
  "datePublished": "2025-09-30",
  "dateModified": "2025-09-30",
  "keywords": "vLLM, Ollama, SGLang, LLaMA.cpp, LLM Serving, PagedAttention, GGUF, Inference, 高吞吐, 推理引擎"
}
</script>