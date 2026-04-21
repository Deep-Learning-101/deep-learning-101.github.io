---
layout: default
title: 檢索增強生成 (RAG) 實戰指南 | 本地部署、混合檢索與 Rerank
description: 2026 最新 RAG 技術實戰指南。從零打造高精準度本地端 RAG 系統，涵蓋 LLM 推理框架 (xinference)、數據清洗切塊 (Chunking)、混合檢索與 Qwen3/Gemini Rerank 排名優化技巧。
permalink: /RAG
lang: zh-Hant
schema_type: article
---

{% include header.html %}

---

{% include ai-share.html %}

---

# 從零到一：打造本地端高精準度 RAG 系統的實戰指南 (涵蓋環境部署、數據處理、混合檢索與 Rerank)

> **🚀 本文重點摘要 (TL;DR)：**
> **RAG (檢索增強生成)** 是一種結合外部知識庫檢索與生成式 AI 的技術，能有效解決 LLM 的幻覺問題。
> 本文提供從零打造高精準度 RAG 系統的實戰指南，涵蓋 **環境部署**、**數據清洗**、**Chunk**、**混合檢索 (Hybrid Search)** 與 **重排序 (Rerank)** 的關鍵技巧。

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**日期**：2026年04月21日 <> 2026年01月02日 <> 2025年07月30日 <> 2024年7月7日  
**相關文章 I**：2024-07-07：[檢索增強生成 (Retrieval-Augmented Generation, RAG) 不是萬靈丹：檢索增強生成的挑戰與優化技巧](https://blog.twman.org/2024/07/RAG.html)  
**相關文章 II**：2025-07-16：[臺灣大型語言模型及文字嵌入和重排序模型性能評測與在地化策略分析報告](https://deep-learning-101.github.io/Blog/TW-LLM-Benchmark)  
**相關文章 III**：2026-04-21：[Sovereign Heuristic Intelligence & Enterprise Logic Defense (主權啟發式情資與企業邏輯防禦系統)](https://deep-learning-101.github.io/SHIELD/)

---

<h2 id="toc">目錄</h2>

- [文章概述](#overview)
- [為何 RAG？](#why-rag)
- [一：基礎環境部署](#env-setup)
  - [選擇本地端推理框架](#local-inference)
- [二：RAG 優化核心流程](#rag-pipeline)
  - [資料準備與嵌入 (Data Preparation & Embedding)](#data-prep)
    - [2.1 Chunking 觀念建立：垃圾進，垃圾出](#chunking)
    - [2.2 高品質數據提取與處理工具](#data-tools)
    - [2.3 選擇合適的嵌入模型 (Embedding)](#embedding)
  - [資料檢索 (Data Retrieval)](#retrieval)
    - [2.4 檢索策略：為何需要混合檢索 (Hybrid Search)？](#hybrid-search)
  - [檢索後處理 (Post-Retrieval Processing)](#post-retrieval)
    - [2.5 Rerank：從「找得全」到「選得準」的關鍵一步](#rerank)
  - [前沿架構突破 (Advanced Paradigm)](#advanced-paradigm)
    - [2.6 無向量 RAG：PageIndex 樹狀推理](#pageindex)
    - [2.7 突破表格與排版限制：無向量視覺檢索 (Vectorless Visual RAG)](#visual-rag)
  - [LLM 生成優化 (LLM Generation)](#llm-gen)
  - [迭代優化與評估 (Iterative Optimization & Evaluation)](#evaluation)
- [總結](#summary)

---

<h2 id="overview">文章概述</h2>

分享在實作 RAG（Retrieval-Augmented Generation）過程中遇到的挑戰與優化技巧，並強調 RAG 並非萬靈丹，需根據實際需求進行適當的設計與調整。

---

<h2 id="why-rag">為何 RAG？</h2>

- **聚焦私有資料**：RAG 能專注於私有資料的檢索與生成，提升回應的相關性。
- **結合檢索與生成**：透過檢索相關資料並結合生成模型，提供更精確的答案。

RAG 提供了一種結合檢索與生成的強大方法，但並非適用於所有情境。實作時需根據實際需求選擇合適的工具與策略，並注意資料處理與模型部署的細節，才能發揮其最大效益。

<h2 id="env-setup">一：基礎環境部署</h2>

<h3 id="local-inference">選擇本地端推理框架</h3>

想在自己的本地端跑大模型，首先需要部署一套推理框架。常見的選擇有 [`Ollama`](https://ollama.com/) 、[VLLM](https://github.com/vllm-project/vllm)和 [`xinference`](https://github.com/xorbitsai/inference)。
`Ollama` 的安裝和執行非常簡單，而 `xinference` 依個人體驗在管理多模型和多卡並行上提供了更大的彈性與便利性，對於進階使用者來說可能是更方便的選擇。

> **補充觀點：為何選擇本地端部署？**
> 選擇在本地部署模型，不僅是為了探索技術，更是一種在**成本、執行速度和數據隱私**三者之間進行權衡的策略。本地化意味著對數據有完全的掌控權，並能避免 API 呼叫的延遲和費用。

**安裝與啟動範例 (以 xinference 為例):**

```bash
# 升級或安裝 xinference，包含所有依賴項
pip install --upgrade "xinference[all]"

# 在指定的 GPU (例如 1, 2, 3 號卡) 上啟動服務，並監聽所有 IP
CUDA_VISIBLE_DEVICES=1,2,3 xinference-local -H 0.0.0.0 -p 6006
```

---

<h2 id="rag-pipeline">二：RAG 優化核心流程</h2>

<h3 id="data-prep">資料準備與嵌入 (Data Preparation & Embedding)</h3>

<h4 id="chunking">2.1 Chunking 觀念建立：垃圾進，垃圾出</h4>

> 在建立知識庫時，最忌諱的就是直接將原始文件一股腦地丟進系統；**請千萬不要無腦的塞入文檔讓它自動切割！**
> 因為 RAG 系統的基礎是高質量的知識區塊 (Chunk)。如果分塊不佳、內容雜亂或包含大量無關資訊，後續的檢索模型再強大，也無法從一堆「垃圾」中準確找出黃金。

**💬 業界實戰痛點：文件切割時，如何規避語意被切斷的問題？**

**語意截斷核心問題：**
當文件內容被機械切割成固定大小的 chunk（塊）時，一個完整語意的單元可能被拆成兩半。例如：
*「企業用戶享有優先客服通道，響應時間不超過 2 小時，並可申請專屬技術顧問服務。」*
若切割邊界落在「2 小時，」處，前半句和後半句各自成為獨立的 chunk，單獨向量化後語意殘缺，檢索時因相關性不足均未被召回，導致關鍵資訊「消失」。

基礎的 **重疊切割（Overlap）** 方案僅能保證跨邊界的文字不丟失，但無法解決「完整語意被拆散後每一半都不夠強」的問題；而單純增大 chunk size 則會引入雜訊，降低檢索精度。

**📋 實戰六大方案完整匯整：**

  * **✅ 方案一：重疊切割（Overlap）——基礎兜底**
      * **核心原理**：相鄰 chunk 保留一段重疊內容（推薦 chunk_size 的 10%～20%），確保跨邊界的連續文字至少完整出現在一個 chunk 中。
      * **優點**：實作簡單、成本可控（儲存增加約 20%）、適用所有文件類型。
      * **⚠️ 局限**：僅為概率性保護，無法識別真正語意邊界，長句仍可能被截斷，需搭配其他策略。
  * **✅ 方案二：按語意邊界切割（Semantic Boundary Chunking）**
      * **核心原理**：以自然語意單位（句子、段落）為邊界切割，避免在句子中間截斷。
      * **實作方式**：使用 NLP 工具（如 spaCy, NLTK）識別句子邊界，並以句子為單位填充 chunk 直至達到大小上限。進階做法可針對論文等內容優先按段落切割。
      * **優點**：chunk 語意完整、向量化更準確、檢索質量顯著提升。
      * **⚠️ 局限**：chunk 大小不均勻，需設上限兜底；複雜句式可能識別錯誤。
  * **✅ 方案三：句子視窗檢索（Sentence Window Retrieval）**
      * **核心邏輯**：將「切割」與「檢索回傳」分離——存儲時按單句切割並獨立向量化，檢索命中後回傳該句的「前後 N 句」形成上下文視窗給 LLM。
      * **優點**：檢索粒度細（精度高、噪聲少），生成時上下文完整。
      * **⚠️ 局限**：存儲量巨大（記錄數≈文檔句子總數）、向量計算成本增加。
  * **✅ 方案四：父子切割（Parent-Child Chunking）**
      * **核心邏輯**：同份內容存兩份——子 chunk（小粒度，如 200 token）用於精準檢索，父 chunk（大粒度，如 1000 token）用於生成，兩者通過 ID 關聯。
      * **優點**：檢索精準，且生成時具備大塊上下文，靈活性高。
      * **⚠️ 局限**：存儲翻倍、索引結構較複雜。
  * **✅ 方案五：命題化切割（Propositions-based Chunking）**
      * **核心原理**：用 LLM 將文件分解為獨立「命題」（Proposition）——每個命題是一個自包含的陳述句，含完整主賓語、無上下文依賴。
      * **優點**：語意密度最高、檢索精度極佳、向量化效果最好。
      * **⚠️ 局限**：需額外呼叫 LLM，成本高昂且速度慢，適合醫療/金融等高質量要求場景。
  * **✅ 方案六：上下文檢索 Contextual Retrieval（Anthropic 2024）**
      * **核心問題解決**：孤立 chunk 向量化時丟失全局語境（如「此條款自 2024 年生效」不知是哪個條款）。
      * **操作步驟**：用 LLM 讀取完整文檔，為每個 chunk 生成 1–2 句背景說明，然後將 Context + chunk 整體做 Embedding/BM25 索引。
      * **優點**：結合 BM25 混合檢索，Top-20 檢索失敗率降低約 49%。利用 Prompt Caching 機制可大幅降低成本。
      * **⚠️ 局限**：仍需消耗一定的 LLM Token 成本。

**🎯 工程實戰：方案選型策略**

| 方案 | 核心思路 | 適用場景 | 代價 |
| :--- | :--- | :--- | :--- |
| **重疊切割** | 相鄰 chunk 內容重疊 | 所有場景（基礎兜底） | 存儲輕微增加 |
| **語意邊界切割** | 按句子/段落邊界切 | 段落清晰的文件（論文、技術文檔） | 需 NLP 工具 |
| **句子視窗檢索** | 細粒度檢索 + 動態擴展上下文 | 追求高召回精度 | 存儲量大 |
| **父子切割** | 小塊檢索、大塊生成 | 通用場景（效果均衡） | 存儲翻倍、索引複雜 |
| **命題化切割** | LLM 分解為獨立命題 | 高質量知識庫（醫療/金融） | LLM 成本高 |
| **Contextual Retrieval** | 向量化前補全 chunk 背景 | 語境強、chunk 孤立問題嚴重 | LLM 成本（緩存可降） |

> **💡 工程組合建議**：
> 預設方案：重疊切割（兜底）+ 語意邊界切割（保質量）。
> 高質量需求：疊加父子切割或 Contextual Retrieval。
> 極致精度：命題化切割（用成本換取最高質量）。

<h4 id="data-tools">2.2 高品質數據提取與處理工具</h4>

為了確保輸入資料的品質，我們需要使用專業工具進行精細的文本提取與處理。

* **通用文件解析：**
    * MinerU：一站式開源工具，擅長將 PDF 轉換為結構化的 Markdown。
    * omniparse：支援多種文件格式，從文件到多媒體。
    * unstructured：強大的非結構化數據處理函式庫。
    * PDFlux：高質量的 PDF 解析工具 (閉源)。

* **掃描件/圖片處理 (OCR)：**
    * PaddleOCR：準確率高的多語言 OCR 工具。
    * RapidOCR：輕量且快速的 OCR 函式庫。

* **Python PDF 處理庫：**
    * `PyMuPDF`: 輕量、高效，適合進行底層的 PDF 文字與元素提取。

> **補充觀點：工具的目標**
> 使用這些工具的最終目標是實現**最佳化分塊 (Optimal Chunking)** 和**預處理關鍵資訊**。這意味著我們要將文件切分成有意義、上下文連貫的段落，並清理掉無關的噪聲，確保每個文本片段在被單獨檢索時仍能表達清晰的含義。

<h4 id="embedding">2.3 選擇合適的嵌入模型 (Embedding)</h4>

在大型語言模型（LLM）應用中，當涉及檢索增強生成（Retrieval-Augmented Generation, RAG）時，其核心目標是為 LLM 提供精準且具備上下文的資訊，從而生成高品質、具事實根據的回應。傳統的關鍵字搜尋方法已不足以應對複雜的語義理解需求。為此，RAG 系統引入了嵌入（Embedding）模型和重排序（Reranking）模型，共同構成了高效能資訊檢索的基石；它們直接影響到 RAG 系統檢索資訊的相關性與準確性。

**Embedding (嵌入)**

*   **Embedding 模型（召回階段）**：是將文本轉換為機器能理解的數值向量的過程。這個向量能捕捉文本的語義資訊，使得意思相近的文本在向量空間中也相互靠近。負責將文本（如文件、段落或使用者查詢）轉換為高維向量（即嵌入），這些向量能夠捕捉文本的語義資訊。此階段的主要任務是「召回」（Recall）。系統使用如 `BAAI/bge-m3` 等 embedding 模型，將龐大知識庫中的所有文件與使用者的查詢轉換為高維度的語義向量。透過計算查詢向量與文件向量之間的相似度，能夠快速篩選出語義上最相關的候選文件。這個階段的目標是盡可能地擴大搜尋範圍，確保所有潛在相關的資訊都能被納入初步的候選清單中。RAG 系統的性能嚴重依賴於這些嵌入的品質，因為高品質的嵌入能確保檢索到最相關的內容。

在繁體中文的檢索場景中，一些模型的表現較為突出：

* **開源 (很多人可能不能用的)**： `Qwen3-Embedding`、`BAAI/bge-m3`、`BAAI/bge-large-zh-v1.5`
* **開源 (大多數人都能用的)**：multilingual-e5-large-instruct

> **補充觀點：選擇的重要性**
> 強調**選擇合適的嵌入模型**是優化檢索品質的關鍵第一步。不同的模型在捕捉語義特徵上有各自的強項和偏好，選對模型能讓你的檢索任務事半功倍。

評估嵌入模型品質的標準基準測試是 MTEB (Massive Text Embedding Benchmark)。

*   **[MTEB (Massive Text Embedding Benchmark)](https://huggingface.co/spaces/mteb/leaderboard)**: MTEB 是一個大規模、多任務、多語言的 embedding 模型評測基準，已成為業界標準。它涵蓋8種嵌入任務，包括位元組挖掘、分類、聚類、配對分類、重排序、檢索、語義文本相似度（STS）和摘要，橫跨181個數據集、多個領域、文本長度和語言。
    *   **檢索 (Retrieval)**: 評估模型為給定查詢找到相關文件的能力，是 RAG 應用中最關鍵的指標。一個廣泛使用的指標是 **NDCG@10 (Normalized Discounted Cumulative Gain @ 10)**，它評估前10個檢索結果的品質，考慮到結果的相關性及其在列表中的位置，值介於0到1之間，1表示完美匹配。
    *   **語義文本相似度 (Semantic Textual Similarity, STS)**: 衡量模型判斷兩個句子語義相似度的能力，使用斯皮爾曼等級相關係數（Spearman correlation）評分。
    *   **分類 (Classification)**: 測試模型提取的特徵向量是否適用於下游的分類任務，通常使用 F1 分數作為指標。
    *   **聚類 (Clustering)**: 評估模型將相似文件分組的能力，使用 v-measure 評分。

*   **[C-MTEB (Chinese MTEB)](https://pypi.org/project/C-MTEB/#leaderboard:~:text=(model)-,Leaderboard,-1.%20Reranker)**: 儘管 MTEB 涵蓋多種語言，但要精準評估模型在特定語言文化下的表現，仍需本地化的評測集。C-MTEB 正是為此而生，它是一個專門針對中文 embedding 模型的評測基準，包含了 35 個中文數據集，涵蓋了與 MTEB 類似的任務類型。C-MTEB 的推出及其被整合至主流排行榜，凸顯了本地化評測對於開發高水準區域語言模型的重要性。

多種嵌入模型被廣泛用於RAG系統。截至2025年中，此領域的競爭已進入白熱化階段，MTEB 全球排行榜的頂端由 Google 和阿里巴巴的最新模型佔據，過去的領先者如 BAAI 的 BGE 系列、Microsoft 的 E5 系列等則面臨激烈挑戰。

1.  **[Google Gemini Embedding (當前榜首)](https://ai.google.dev/gemini-api/docs/embeddings?hl=zh-tw)**:
    * **gemini-embedding-001**: Google 推出的此模型在發布後迅速登上 MTEB 排行榜首位，展現了其最先進（State-of-the-Art）的文本表徵能力。作為一個閉源商用模型，它在各項評測中（檢索、分類、聚類等）取得了極高的綜合平均分，使其成為追求極致性能、且在 Google Cloud 生態內的開發者的首選。

2.  **[Alibaba Qwen3 Embedding (開源領頭羊)](https://qwenlm.github.io/zh/blog/qwen3-embedding/)**:
    * **Qwen3-Embedding 系列 (0.6B, 4B, 8B)**: 這是由 Qwen 團隊基於強大的 Qwen3 基礎模型訓練的新一代 Embedding 系列。根據其官方報告，**`Qwen3-Embedding-8B`** 模型在發布時曾一度登頂 MTEB 多語言榜單，目前也以微弱差距緊隨 `gemini-embedding-001` 之後，位居第二，是**開源模型中的 undisputed champion (無可爭議的冠軍)**。
    * **核心優勢**:
        * **卓越性能與泛化性**: 繼承了 Qwen3 的多語言理解能力（支援超過100種語言），在 MTEB 和 C-MTEB 上均表現頂尖。
        * **靈活架構**: 提供從 0.6B 到 8B 的多種尺寸，並支援**自訂輸出維度 (MRL Support)** 和 **指令微調 (Instruction Aware)**，讓開發者能根據成本和效能需求進行客製化，極具彈性。
        * **先進的訓練方法**: 採用了創新的三階段訓練範式，特別是利用 Qwen3 自身生成能力來建構大規模弱監督訓練資料，突破了傳統方法的限制。

3.  **昔日強者與現存勁旅**:
      * **BAAI/bge-m3 & JinaAI-v2-base-en**: 這些模型曾經是 MTEB 排行榜上的佼佼者，但隨著新模型的推出，其排名已有所下滑。儘管如此，`bge-m3` 憑藉其獨特的多向量檢索能力和長文本支援，在特定場景下依然有其價值。它們的存在證明了這個領域技術迭代的速度之快。
      * **Voyage AI & NV-Embed**: 這些同樣是性能非常強勁的（商用）模型，雖然被最新的 Gemini 和 Qwen3 超越，但依然處於排行榜的頂級梯隊中，是特定需求下的可靠選項。
      * **intfloat/multilingual-e5-large-instruct**: 這是由 Microsoft Research 推出的 E5 系列中的重要多語言模型。E5 系列是推廣**指令微調 (Instruction Tuning)** 於 Embedding 領域的先驅之一，其設計理念對後續許多模型產生了深遠影響。雖然其性能已被新一代模型超越，但它仍然是一個非常穩固的開源基準模型，廣泛應用於學術研究和業界實踐中。

**表 關鍵 Embedding 模型特性比較**

| 模型名稱 | 主要語言 | 最大上下文長度 (Tokens) | MTEB Score (Avg) | C-MTEB Score | 關鍵優勢與表現摘要 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **google/gemini-embedding-001** | 多語言 | 8192 | **68.61** | 71.04 | 閉源商用，性能頂尖，生態整合。**MTEB 全球排行榜當前 \#1**。 |
| **Alibaba-NLP/Qwen3-Embedding-8B** | 多語言 (100+) | 32768 | **68.12** | 72.88 | 開源，性能頂尖，架構靈活，可調維度。**MTEB 全球排行榜 \#2，開源模型 \#1**。 |
| **Alibaba-NLP/Qwen3-Embedding-4B** | 多語言 (100+) | 32768 | **66.86** | 71.85 | Qwen3 系列中型模型，高效能。MTEB 排名頂尖，具備成本效益。 |
| **voyage-ai/voyage-large-2-instruct** | 多語言 | 16384 | **66.08** | 68.32 | 閉源商用，檢索性能強勁。曾為 MTEB 榜首，現仍居頂級梯隊。 |
| **BAAI/bge-m3** | 多語言 (100+) | 8192 | 64.63 | 68.31 | 多向量檢索，長文本處理，多功能。排名已下滑，但在特定功能上仍具優勢。 |
| **intfloat/multilingual-e5-large-instruct** | 多語言 | 512 | 62.13 | 62.91 | 開源，指令微調先驅，穩定的基準模型。經典模型，已被新模型超越。 |
| **JinaAI/jina-embeddings-v2-base-en** | 英文為主 | 8192 | 61.15 | N/A | 曾是強力的開源選項。排名已下滑，被新模型大幅超越。 |

*(註：MTEB/C-MTEB 分數是浮動的，數據基於 [2025 年 Q3 的 CSV 檔案](https://huggingface.co/spaces/mteb/leaderboard)。N/A 表示無適用的公開分數。)*

<h3 id="retrieval">資料檢索 (Data Retrieval)</h3>

<h4 id="hybrid-search">2.4 檢索策略：為何需要混合檢索 (Hybrid Search)？</h4>

檢索是從向量資料庫中找出與使用者問題相關資訊的過程。常見的策略包括：

* **向量檢索：** 透過計算查詢向量與文檔向量的相似度，找出語義上最接近的內容。
* **全文檢索：** 基於關鍵字索引，找出精確匹配詞彙的文件。
* **TopK：** 召回與問題相似度最高的 K 個文件片段。
* **Score 閾值：** 只召回相似度分數超過特定門檻的文件片段。

模型選擇的決策比以往任何時候都更加關鍵，需要綜合考量性能、成本、開源與否以及特定場景需求。

* **追求極致性能的閉源方案**: 若預算充足且追求當前最高性能，`google/gemini-embedding-001` 是 MTEB 榜單上的冠軍選擇。
* **追求頂級性能的開源方案**: `Alibaba-NLP/Qwen3-Embedding-8B` 是目前開源社群的性能天花板，尤其適合需要處理中、英文及多語言混合內容的 RAG 系統。其靈活的架構（可調維度、指令適配）也為進階優化提供了可能。
* **專注於中文的應用**: 根據 C-MTEB 排行榜，`Qwen3` 系列在中文任務上同樣表現優異。與 `BAAI/bge-large-zh-v1.5` 這類專為中文設計的經典模型相比，`Qwen3` 提供了更強的綜合性能和多語言兼容性，可能是更現代的選擇。
* **考量特定功能的舊有模型**: 如果 RAG 系統有特殊需求，例如 `BAAI/bge-m3` 的多向量（密集+稀疏）檢索能力，那麼即便其綜合排名下滑，仍可能因其獨特功能而入選。

> **補充觀點：為何需要混合檢索 (Hybrid Search)？**
> 單一的檢索方式存在盲點：向量檢索可能忽略關鍵字，而全文檢索無法理解語義。**混合檢索**將兩者結合，它既能透過全文檢索確保**精確匹配**不遺漏，又能透過向量檢索找到**語義相關**的內容，從而**大幅提升覆蓋率 (Recall)**，是目前最主流且效果最好的檢索策略。

<h3 id="post-retrieval">檢索後處理 (Post-Retrieval Processing)</h3>

<h4 id="rerank">2.5 Rerank：從「找得全」到「選得準」的關鍵一步</h4>；[更多 Embedding和Rerank模型說明在這](#Appendix-Embedding-Reranking-RAG)

初步檢索（尤其是`混合檢索`）的目標是「找得全」，但這也意味著結果中可能混雜著一些相關性不高的內容。這時就需要 Rerank 來進行「二次精選」。

在初步檢索之後，Reranker 模型是提升 RAG 系統回應品質的第二道關鍵防線。

Reranker 模型的核心是其 cross-encoder 架構。與 embedding 模型（bi-encoders）分別為查詢和文件生成獨立的向量不同，cross-encoder 將「查詢」和「單一候選文件」作為一個整體同時輸入模型進行處理。這種設計允許模型在內部對查詢和文件的每一個 token 之間進行深度、細粒度的注意力計算，從而給出一個極其精準的相關性分數。

這種高精準度的代價是計算量遠大於 bi-encoder，因此它不適合用於對整個龐大知識庫進行全面篩選，而是作為「精煉器」，僅對由 embedding 模型快速召回的前 k 個（例如前 20-50 個）最相關的候選文件進行重新排序。

常見的評估指標包括**命中率（Hit Rate）和平均倒數排名（MRR, Mean Reciprocal Rank）**。研究顯示，優秀的重排序模型能持續提升幾乎所有嵌入模型的這兩項指標。

根據現有研究，市場上主流的 Reranker 模型包括 `BAAI/bge-reranker-v2-m3`、Jina AI 的 `jina-reranker-v2-base-multilingual` 以及由阿里巴巴開發的 `Qwen3-Reranker` 系列。一份關鍵的評測報告對這些模型在多個檢索相關基準上的表現進行了比較，包括 MTEB-R（英文檢索）、CMTEB-R（中文檢索）、MMTEB-R（多語言檢索）和 MLDR（多語言長文件檢索）。

*   **重排序模型（精煉階段）**: 在初始檢索步驟之後，當召回的文件數量眾多、包含雜訊或與查詢意圖不夠一致時，重排序模型會介入。此階段扮演了至關重要的「精煉」（Precision）角色。它們使用更複雜的模型（例如交叉編碼器 Cross-Encoder，如 `BAAI/bge-reranker-v2-m3`）來重新排序或過濾這些文件，以提高其相關性。Cross-encoder 會將查詢與每一份候選文件成對地輸入模型，進行深度的互動式比對與注意力計算。這種方法的計算成本較高，但能極其精準地評估文件與查詢的真實關聯性。透過此步驟，系統能確保最終傳遞給 LLM 的上下文是關聯性最強、最精準的資訊，從而大幅提升生成回應的準確性與事實一致性。這對於處理時間敏感的即時資訊尤其重要。

**Embedding 快篩 vs. Reranker 精排：**
嵌入模型適合快速從海量資料中找出「可能相關」的候選文件，但它無法完全抓住查詢和文件之間的細微差異。而 Rerank 模型則能更深入地分析每個候選文件的內容，進行更精確的相關性排序。兩者搭配，既保證了速度，也提高了最終結果的精確度。

> **補充觀點：Rerank 的價值**
> 如果說初步檢索是為了**「找得全」(High Recall)**，那麼 Rerank 的核心任務就是**「選得準」(High Precision)**。它像一位嚴格的評審，確保最終交給 LLM 生成答案的，是最高度相關的**「黃金上下文」**。

在精煉階段，Reranker 模型的角色至關重要。近年來，[`Alibaba-NLP/Qwen3-Reranker`](https://qwenlm.github.io/zh/blog/qwen3-embedding/) 系列的發布，**幾乎重新定義了 Reranker 模型的性能標竿**。

數據評測（如下表所示）清晰地揭示了 `Qwen3-Reranker` 的統治力。無論是在英文檢索（MTEB-R）、中文檢索（CMTEB-R）、多語言檢索（MMTEB-R），甚至是程式碼檢索（MTEB-Code）任務上，`Qwen3-Reranker` 的 4B 和 8B 版本都取得了遠超 `BGE-reranker-v2-m3`、`jina-reranker-v2-base-multilingual` 等前代模型的成績。

  * **Qwen3-Reranker-4B 和 Qwen3-Reranker-8B** 不僅在傳統文本檢索上表現優異，在程式碼相關的檢索任務上也大幅領先，這顯示了 Qwen3 基礎模型強大的通用語義理解能力。對於任何希望將 RAG 系統檢索精度推向極致的應用，`Qwen3-Reranker` 系列已成為不二之選。
  * **jina-reranker-v2-base-multilingual** 是 Jina AI 繼其 Embedding 模型後推出的高效能多語言重排序模型。它支援英文、中文、西班牙文等多種語言，並在 8K 的長上下文處理上表現出色，使其在處理長文件檢索時具有優勢。雖然在基準評測上已被 Qwen3 系列超越，但其在多語言長文本場景的專注設計，使其在特定應用中仍具競爭力。

| Model | Param | MTEB-R | CMTEB-R | MMTEB-R | MLDR | MTEB-Code | FollowIR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-Embedding-0.6B | 0.6B | 61.82 | 71.02 | 64.64 | 50.26 | 75.41 | 5.09 |
| **jina-reranker-v2-base-multilingual** | **0.3B** | **58.22** | **63.37** | **63.73** | **39.66** | **58.98** | **-0.68** |
| gte-multilingual-reranker-base | 0.3B | 59.51 | 74.08 | 59.44 | 66.33 | 54.18 | -1.64 |
| BGE-reranker-v2-m3 | 0.6B | 57.03 | 72.16 | 58.36 | 59.51 | 41.38 | -0.01 |
| Qwen3-Reranker-0.6B | 0.6B | 65.80 | 71.31 | 66.36 | 67.28 | 73.42 | 5.41 |
| Qwen3-Reranker-4B | 4B | 69.76 | 75.94 | 72.74 | 69.97 | 81.20 | 14.84 |
| Qwen3-Reranker-8B | 8B | 69.02 | 77.45 | 72.94 | 70.19 | 81.22 | 8.05 |

*(註：排序結果基於Qwen3-Embedding-0.6B的top-100向量召回結果進行排序)*

數據明確顯示了重排序模型在優化搜索結果方面的顯著性。幾乎所有嵌入模型都透過重排序獲得了改進。重排序模型，特別是 `CohereRerank` 和 `bge-reranker-large` (或其更新版本如 `BGE-reranker-v2-m3`)，展現了將任何嵌入模型轉化為具有競爭力的模型的能力。

然而，引入重排序模型會增加延遲和系統複雜性。儘管開箱即用的重排序模型在某些推理任務上可能表現不佳，但透過微調可以實現最先進的性能。這也顯示了重排序模型在真實世界應用中，需要在模型大小、排名準確性以及延遲/吞吐量等系統要求之間取得平衡。

<h3 id="advanced-paradigm">前沿架構突破 (Advanced Paradigm)</h3>

<h4 id="pageindex">2.6 無向量 RAG：PageIndex 樹狀推理</h4>

**📌 傳統向量 RAG 處理長文檔的三大缺陷：**

1.  **語意相似 ≠ 相關**：敘述性文字與具體數據的語意可能極度接近，但用途完全不同（例如 CEO 致辭 vs. 真實資產負債表），導致關鍵數據被忽略。
2.  **分塊破壞結構**：諸如「如表3.2所示」的文字與實際表格經常被強制拆分至不同區塊，導致引用完全失效。
3.  **意圖與表述不對齊**：用戶問題（如「總負債」）與文檔內部寫法（如「流動負債」「長期債務」）的措辭不一致，導致餘弦相似度無法匹配。

**🔄 新思路：PageIndex 仿生學設計**
PageIndex 不將文檔硬切碎轉為向量，而是**模擬人類分析師閱讀長文檔的行為**：`先看目錄 → 判斷章節 → 翻至對應頁面 → 若錯誤則回溯`。它將文檔構建為**樹狀結構**，讓 LLM 直接在樹上進行推理導航。

**⚙️ 運作機制與關鍵工程細節**

1.  **建構目錄樹**：LLM 解析文檔標題與章節，生成層級樹。每個節點包含標題、頁碼、摘要與關鍵主題。系統具備三路徑自動降級機制（最優路徑直接解析、次優路徑掃描正文、兜底路徑「發明目錄」），確保無結構文檔也能運行。
2.  **推理式搜尋（迭代循環）**：
      - LLM 根據 Query 判斷可能章節（例如：長期負債 → 財務報表 → 附註）。
      - 展開節點提取原始文字。
      - LLM 自行評估資訊是否足夠：✅ 足夠 → 產出答案（附頁碼）；❌ 不足 → 返回目錄樹選擇下一個節點繼續找。
3.  **動態容錯**：遇到目錄頁碼與實體 PDF 頁碼不符時，系統會計算偏移量並全域校正；建樹後也會併發驗證，避免 LLM 幻覺。

**📊 傳統 RAG vs. PageIndex 對比**

| 維度 | 傳統向量 RAG | PageIndex (無向量) |
| :--- | :--- | :--- |
| **檢索方式** | 向量相似度搜尋 | LLM 推理式樹搜尋 |
| **索引方式** | Embedding + 向量資料庫 | 層級樹結構（JSON） |
| **分塊策略** | 固定大小/語意分塊 | 保留文檔自然邊界與結構 |
| **準確率** (FinanceBench) | ~50% | **98.7%** |
| **可解釋性** | 低（僅提供相似度分數） | **高**（完整推理鏈 + 頁碼引用） |
| **延遲** | 低（單次快速查詢） | 較高（需多次 LLM 推理呼叫） |
| **適用場景** | 大規模文檔集合快速找尋 | **單篇複雜長文檔精確問答** |

> **🏗️ 最佳實踐：混合架構推薦**
> 由於 PageIndex 需要多次 LLM 推理，延遲較高，實戰中建議採取混合架構：
>
>   * **粗篩（向量檢索）**：用傳統 Embedding 搜尋從大量文檔中快速鎖定目標文檔。
>   * **精查（PageIndex）**：針對鎖定的目標文檔啟動樹推理，精確提取答案與頁碼出處。非常適合金融報告、法律合同、學術論文等需嚴格追蹤引用的場景。

**💻 簡化實作步驟參考 (以 Gemini API 為例)**

<details>
<summary>👉 點擊展開：Python 實作概念碼</summary>

```python
import fitz
import json
from google import genai

# 1. 解析文檔 (PyMuPDF)
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page_num": i+1, "text": text})
    return pages

# 2. 按章節分組 (保留自然邊界)
def group_pages_into_sections(pages, per_section=3):
    sections = []
    for i in range(0, len(pages), per_section):
        batch = pages[i:i+per_section]
        section_id = f"S{str(i//per_section+1).zfill(3)}"
        combined_text = "nn".join(p["text"] for p in batch)
        sections.append({
            "section_id": section_id,
            "start_page": batch[0]["page_num"],
            "end_page": batch[-1]["page_num"],
            "text": combined_text
        })
    return sections

# 3. 建樹索引 (LLM 生成摘要)
def index_section(section, client):
    preview = section["text"][:1500]
    prompt = f"""Read this section... Respond with ONLY valid JSON: {{"title":"...","summary":"...","key_topics":[...]}}"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    parsed = json.loads(response.text.strip())
    return {
        "node_id": section["section_id"],
        "title": parsed["title"],
        "pages": f"{section['start_page']}-{section['end_page']}",
        "summary": parsed["summary"],
        "key_topics": parsed["key_topics"]
    }

# 4. 樹搜尋 (LLM 推理選節點)
def retrieve_sections(tree, query, client):
    prompt = f"""You are a document retrieval expert... Respond with ONLY valid JSON: {{"reasoning":"...","selected_ids":[...],"confidence":"..."}}"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return json.loads(response.text.strip())

# 5. 生成答案
def generate_answer(query, context, client):
    prompt = f"""Answer using only context. Be specific. Cite page numbers.nCONTEXT:{context}nQUESTION:{query}nANSWER:"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text.strip()
```

</details>

<br>

<h4 id="visual-rag">2.7 突破表格與排版限制：無向量視覺檢索 (Vectorless Visual RAG)</h4>

在企業級知識庫（如財務報表、資安法規、醫療 SOP）中，充滿了大量的**表格、流程圖與複雜排版**。傳統 RAG 系統在「文字提取」階段，往往會將 2D 的表格壓扁成 1D 的純文字，導致欄位錯位、語意斷裂，進而引發嚴重的 LLM 幻覺。

即便使用了前面提到的 PageIndex 樹狀推理，如果最終餵給 LLM 的依然是「轉換後的純文字」，在解讀複雜圖表時依然會遇到瓶頸。為此，結合多模態大模型（如 Gemini 1.5/2.0 Pro、GPT-4o）的 **「無向量視覺檢索 (Visual RAG)」** 成為了終極解決方案。

**🚀 核心架構：結構化目錄 + 原始影像直讀**

此架構完全捨棄了「文字切塊 (Chunking)」與「向量化 (Embedding)」，轉而模擬人類「查閱參考書」的真實行為：

1. **結構解析與座標映射 (無 LLM 延遲)**：
   放棄傳統的 PDF 轉文字，改用底層物理結構解析工具（如 `OpenDataLoader` 搭配 `PyMuPDF`）。系統會快速掃描 PDF，辨識出標題 (Heading)、表格 (Table) 與圖片 (Picture) 的精準座標 (Bounding Box) 與絕對頁碼，藉此建立一棵極輕量級的 JSON 樹狀目錄。這個過程無需呼叫 LLM，建檔速度極快。
2. **總圖書館長定位 (目錄檢索)**：
   當收到使用者查詢時，系統不搜尋全文，而是將使用者的問題與「JSON 目錄樹」送給 LLM（擔任總圖書館長的角色）。LLM 透過目錄層級，精準推導出答案位於「哪一份文獻的第 X 頁」。
3. **多模態視覺直讀 (Vision RAG)**：
   鎖定頁碼後，系統**不提取文字**，而是直接從資料夾中調用該頁面的**高畫質原始截圖 (JPEG/PNG)**，將圖片連同使用者的問題直接送給多模態 LLM。

**📊 Visual RAG 的決定性優勢**

* **0% 排版遺失**：無論是跨頁表格、密集的財報數據、或是帶有箭頭指示的 IT 系統架構圖，AI 都是「看著原圖」作答，徹底消滅了文字轉換過程中的結構破壞。
* **解決「文字/圖像」混合難題**：傳統 RAG 面對「請參考圖 3-1 並結合表 2 的數據」這類問題會直接當機，而 Visual RAG 能像人類一樣同時綜整圖文資訊。
* **極低的 Token 浪費**：搜尋階段只需閱讀輕量的 JSON 目錄，生成階段只需傳入單張圖片，大幅降低了將整份長文本塞入 Context Window 的成本。

> **💡 工程實戰建議**：
> 要實作此架構，強烈建議使用 `PyMuPDF` 在建檔期預先將 PDF 所有頁面渲染為高畫質圖片存檔。當檢索命中時，直接利用文件 ID 與頁碼拼接出圖片路徑，即可達成毫秒級的影像調用，完美銜接多模態 LLM 的輸入 API。

**🔍 深度解析：底層數據處理的典範轉移 (Text Stream vs. Spatial Layout)**

要實現完美的無向量視覺檢索，核心前提是必須捨棄傳統的「純文字解析器」。以傳統 LLM 索引 (如舊版 PageIndex) 與新一代物理版面解析 (如 OpenDataLoader, ODL) 為例，兩者在原生數據處理上有著本質的差異：

1. **物理版面感知 vs. 純文字流**：
   * **傳統解析**：將文件視為 1D 的文字麵條。掃描時由左至右、由上至下硬拉出文字，遇到「雙欄排版」或「表格」時，文字順序會大亂，表格也會被壓扁成毫無意義的文字堆。
   * **物理版面解析**：將文件視為 2D 空間地圖。解析器會先看懂版面區塊，圈出「這是一個表格」、「這是一段標題」的精準 Bounding Box 與絕對頁碼，完美保留原生版面結構。
2. **LLM 依賴時機與建檔成本**：
   * **傳統解析**：因為抽出的純文字缺乏結構，必須強制呼叫 LLM 閱讀每一段文字來「腦補」生成摘要與目錄樹，導致建檔過程極度緩慢且 API 成本高昂。
   * **物理版面解析**：依賴底層電腦視覺與啟發式演算法，**建檔過程零 LLM 延遲**。瞬間將物理結構轉為 JSON 目錄，把 LLM 算力 100% 保留到檢索發問的那一刻才使用。
3. **多模態內容的存亡**：
   * **傳統解析**：PDF 裡的架構圖、數學公式通常會變成隱形或亂碼，圖文資訊在建檔瞬間就被抹殺。
   * **物理版面解析**：原生標記出 `<picture>`、`<table>` 等錨點，允許系統精準截取該區塊的高畫質圖片，讓多模態大模型在檢索時得以「看圖說故事」。
4. **目錄樹的真實性 (Truthfulness)**：
   * **傳統解析**：目錄樹是 LLM 根據文字內容「幻想/總結」出來的，容易產生幻覺。
   * **物理版面解析**：目錄樹 100% 基於原生檔案的字體大小、粗細與縮排所建立，是客觀存在的「物理目錄」，不具備任何幻覺空間，所見即所得。
   
<h3 id="llm-gen">LLM 生成優化 (LLM Generation)</h3>

在 `xinference` 或 `Ollama` 中，不僅檢索與重排模型重要，最終用於生成答案的模型也應根據需求選擇。

如果你的 RAG 流程（從數據處理到 Rerank）已經做得非常好，檢索到的上下文品質極高，那麼有時並不需要動用最強大的生成模型（如 GPT-4 等級）。`Llama-3.1-70B-Instruct`就足以生成優質、準確的答案。這同樣是在**準確性和計算成本**之間做出明智的權衡。

<h3 id="evaluation">迭代優化與評估 (Iterative Optimization & Evaluation)</h3>

建立 RAG 系統並非一勞永逸，它是一個需要持續優化和迭代的過程。當系統上線後，我們需要一套機制來衡量其表現。

業界常見的做法包括：

* **自動化評估：** 使用強大的 LLM（如 GPT-4o）作為「評審」，根據相關性、忠實度、簡潔性等指標，自動評估 RAG 系統的回應品質。
* **評估框架：** 利用 `Ragas` 等開源框架，對 RAG 管道的每個環節（檢索、生成）進行量化評分。
* **使用者回饋：** 建立簡單直觀的回饋機制（如點讚/點踩），收集使用者最直接的體驗，這是發現問題、改進系統最寶貴的資訊來源。

---

<h2 id="summary">總結</h2>

通過將實戰操作融入清晰的理論框架，您建立的 RAG 指南將會：

* **更有條理：** 遵循從「數據準備」到「最終評估」的清晰邏輯線，讓讀者能一步步跟隨。
* **更具深度：** 不僅告訴讀者「用這個工具」，還解釋了「為什麼在這個環節要用這類工具」，提升了文章的理論高度。
* **更加全面：** 補充了「生成模型選擇」和「迭代評估」兩個畫龍點睛的環節，讓整個 RAG 實戰指南更加完整且專業。

就臺灣本土大型語言模型（如 `yentinglin/Llama-3-Taiwan` 系列、`taide/Llama-3.1-TAIDE-LX-8B-Chat`、`MediaTek-Research/Llama-Breeze2` 系列）以及國際知名模型（如 `Qwen` 和 `Llama 3.x` 系列）而言，現有資料主要針對這些 LLM 本身在如 TMLU、TMMLU 等語言理解基準測試上的表現進行評估。

嵌入模型和重排序模型是 RAG 系統中不可或缺的組成部分，它們共同確保了提供給 LLM 的資訊的相關性和準確性。雖然有通用的基準測試（如 MTEB、C-MTEB）和評估方法（如 NDCG@10、Hit Rate、MRR）來評估這些模型，且已證明它們對 RAG 系統性能的關鍵影響，但針對特定 LLM（如臺灣本土模型、Qwen、Llama 3.x 系列）作為獨立嵌入/重排序組件的詳細評比數據，在當前資料中尚不充分。這類數據通常會是更專門化的 RAG 系統組件性能評估研究的範疇，並且需要根據具體的應用場景、知識庫特性（如語言、長度）和系統資源限制（如延遲、計算成本）來進行細緻的選擇與優化。

嵌入模型和重排序模型是 RAG 系統中不可或缺的組成部分... 隨著 `Qwen 3` 系列 和 `Google Gemini` 等新一代模型的出現，MTEB 和相關評測的榜單正在被不斷刷新。這表明模型的能力邊界在持續擴展，但也對開發者提出了更高的要求。

最終，成功的 RAG 系統不再僅僅是選擇某個「最好」的模型，而是一個持續評估、測試和權衡的過程。開發者需要根據具體的應用場景、知識庫特性（語言、領域、長度）、以及系統資源限制（延遲、計算成本），動態地選擇最適合的 Embedding 和 Reranker 組合，才能在資訊檢索的「召回」與「精煉」兩個戰場上都取得勝利。

關於這些特定模型在作為 RAG 系統中的**嵌入模型**或**重排序模型**方面的獨立基準測試結果，目前提供的公開資料並未明確提供詳盡的數據。這可能歸因於以下幾點：

*   **專注點不同**：許多 LLM 本身的基準測試關注於其生成和理解能力，而非其作為嵌入或重排序組件的效能。
*   **語言特異性挑戰**：如越南語資訊檢索領域也面臨缺乏專門針對嵌入和重排序任務的基準測試。臺灣繁體中文環境也可能面臨類似的挑戰，儘管存在多語言嵌入模型（如 `embed-multilingual-v3.0`），但針對臺灣特定語言和文化背景進行優化的嵌入或重排序模型，其獨立評測數據可能需要更深入的專門研究。
*   **整合評估**：RAG 系統的性能通常是各個組件（包括嵌入、檢索、重排序和生成）協同作用的結果。一些研究會評估整個 RAG 流程如何提升 LLM 的整體準確性（例如，在眼科問答中，使用 RAG 包含 Cohere 重排序顯著提升了 GPT-4、Llama-3-70B 等模型的準確度）。
*  **避開有疑率模型**：Embedding model 可考慮 multilingual-e5-large-instruct​，Reranker model 可考慮 jina-reranker-v2-base-multilingual

---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/RAG"
  },
  "inLanguage": "zh-Hant",
  "headline": "從零到一：打造本地端高精準度 RAG 系統的實戰指南",
  "description": "2026 最新 RAG 技術實戰指南，涵蓋環境部署、數據處理、混合檢索、Rerank 優化技巧，以及最前沿的 PageIndex 樹狀推理與無向量視覺檢索 (Visual RAG) 架構深度解析。",
  "image": "https://raw.githubusercontent.com/Deep-Learning-101/deep-learning-101.github.io/refs/heads/main/images/DeepLearning101-LOGO.png",
  "author": {
    "@type": "Person",
    "name": "TonTon Huang Ph.D.",
    "url": "https://twman.org/"
  },
  "publisher": {
    "@type": "Organization",
    "name": "Deep Learning 101, Taiwan",
    "logo": {
      "@type": "ImageObject",
      "url": "https://raw.githubusercontent.com/Deep-Learning-101/deep-learning-101.github.io/refs/heads/main/images/DeepLearning101-LOGO.png"
    }
  },
  "datePublished": "2024-07-07",
  "dateModified": "2026-04-21",
  "articleSection": ["Generative AI", "Information Retrieval", "Machine Learning"],
  "keywords": "RAG, Retrieval-Augmented Generation, LLM, 檢索增強生成, Embedding, Rerank, Qwen3, Gemini, 混合檢索, PageIndex, Visual RAG, 無向量視覺檢索, OpenDataLoader"
}
</script>