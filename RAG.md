---
layout: default
title: "檢索增強生成 RAG LLM 資源懶人包 | Leaderboard, Tools & Papers"
description: "2025 最新 RAG 技術需知 LLM 大語言模型資源彙整。包含環境部署、數據處理、混合檢索與 Rerank 等必讀資源。"
permalink: /RAG
lang: zh-Hant
keywords: ["RAG", "檢索增強生成", "LLM", "LangChain", "Rerank"]
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
**日期**：2026年01月02日 <> 2025年07月30日 <> 2024年7月7日  
**相關文章 I**：[檢索增強生成 (Retrieval-Augmented Generation, RAG) 不是萬靈丹：檢索增強生成的挑戰與優化技巧](https://blog.twman.org/2024/07/RAG.html)  
**相關文章 II**：[臺灣大型語言模型及文字嵌入和重排序模型性能評測與在地化策略分析報告](https://deep-learning-101.github.io/Blog/TW-LLM-Benchmark)


| 🔥 技術傳送門 (Tech Stack) | 📚 必讀心法 (Must Read) |
| :--- | :--- |
| 🤖 [**大語言模型 (LLM)**](https://deep-learning-101.github.io/Large-Language-Model) | 🏹 [**策略篇：企業入門策略**](https://deep-learning-101.github.io/Blog/AIBeginner) |
| 📝 [**自然語言處理 (NLP)**](https://deep-learning-101.github.io/Natural-Language-Processing) | 📊 [**評測篇：臺灣 LLM 分析**](https://deep-learning-101.github.io/Blog/TW-LLM-Benchmark) |
| 👁️ [**電腦視覺 (CV)**](https://deep-learning-101.github.io//Computer-Vision) | 🛠️ [**實戰篇：打造高精準 RAG**](https://deep-learning-101.github.io/RAG) |
| 🎤 [**語音處理 (Speech)**](https://deep-learning-101.github.io/Speech-Processing) | 🕳️ [**避坑篇：AI Agent 開發陷阱**](https://deep-learning-101.github.io/agent) |

**相關文章參考**：
* <b><a href="https://blog.twman.org/2024/09/LLM.html" target="_blank">大型語言模型直接就打完收工？</a></b>：<a href="https://deep-learning-101.github.io/1010LLM">回顧 LLM 領域探索歷程，討論硬體升級對 AI 開發的重要性。</a>
* <b><a href="https://blog.twman.org/2024/07/RAG.html" target="_blank">檢索增強生成(RAG)不是萬靈丹之優化挑戰技巧</a></b>：<a href="https://deep-learning-101.github.io/RAG">探討 RAG 技術應用與挑戰，提供實用經驗分享和工具建議。</a>
* <b><a href="https://blog.twman.org/2024/02/LLM.html" target="_blank">大型語言模型 (LLM) 入門完整指南：原理、應用與未來</a></b>：<a href="https://deep-learning-101.github.io/0204LLM">探討多種 LLM 工具的應用與挑戰，強調硬體資源的重要性。</a>
* <b><a href="https://blog.twman.org/2023/04/GPT.html" target="_blank">解析探索大型語言模型：模型發展歷史、訓練及微調技術的 VRAM 估算</a></b>：<a href="https://deep-learning-101.github.io/GPU">探討 LLM 的發展與應用，硬體資源在開發中的作用。</a>
* <b><a href="https://www.facebook.com/cnanewstaiwan/posts/pfbid02CCrFhyvCcoTmjJaX4aHaSMHmCgnPd1SG21Gbpb4Wo9bgs7QmQArTmhbVPZSLyjrdl" target="_blank">中央社繁體中文預訓練資料集案</a></b>

---

## 文章概述

分享在實作 RAG（Retrieval-Augmented Generation）過程中遇到的挑戰與優化技巧，並強調 RAG 並非萬靈丹，需根據實際需求進行適當的設計與調整。

---

## 為何 RAG？

- **聚焦私有資料**：RAG 能專注於私有資料的檢索與生成，提升回應的相關性。
- **結合檢索與生成**：透過檢索相關資料並結合生成模型，提供更精確的答案。

RAG 提供了一種結合檢索與生成的強大方法，但並非適用於所有情境。實作時需根據實際需求選擇合適的工具與策略，並注意資料處理與模型部署的細節，才能發揮其最大效益。

## 一：基礎環境部署

### 選擇本地端推理框架

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

## 二：RAG 優化核心流程

### 資料準備與嵌入 (Data Preparation & Embedding)

#### 2.1 Chunking  觀念建立：垃圾進，垃圾出

> 在建立知識庫時，最忌諱的就是直接將原始文件一股腦地丟進系統；**請千萬不要無腦的塞入文檔讓它自動切割！**

**為什麼？** 因為 RAG 系統的基礎是高質量的知識區塊 (Chunk)。如果分塊不佳、內容雜亂或包含大量無關資訊（如頁眉、頁腳），那麼後續的檢索模型再強大，也無法從一堆「垃圾」中準確找出黃金，最終只會導致模型產生不準確或無關的答案。


* **考量要素**：將長文件切割成小塊的過程，是 RAG 系統效能優劣的關鍵。其目標是在「保留上下文」與「適應模型處理能力」之間取得平衡。
  * 塊的大小：建議在 100 到 500 個 tokens 之間，過大或過小都會影響檢索與生成效果。
  * 上下文保留：切塊時必須避免破壞句子或段落意義，確保模型能正確理解。
  * 多模態處理：文件可能包含文本、表格、圖片等，需要針對不同內容類型設計切塊策略。
* **切塊策略**
    * **基礎結構化**切塊：
      * 固定大小：最簡單，按固定詞數切割。優點：實現快。缺點：易破壞語義。
      * 句子/段落切塊：按自然語言邊界（句子、段落）切割。優點：語義完整。缺點：塊大小不一，可能超出模型限制。
    * **上下文保留型**切塊：
      * 滑動窗口：讓相鄰的塊有部分重疊。優點：確保上下文連貫。缺點：處理量大，有內容重複。
      * 語義切塊：基於句子語義相似度合併。優點：上下文保留最好。缺點：實現複雜，需依賴語言模型。
    * **結構/內容感知型**切塊：
      * 層級切塊：按章節、段落等文檔結構層次切割。優點：完美保留結構。適用：論文、合同。
      * 內容感知切塊：根據標題、引言、結論等內容特徵動態切換策略。優點：靈活。適用：技術文檔、電子書。
    * **特定內容型**切塊：
      * 表格感知：將表格作為獨立單位處理，保留結構。適用：財務、技術報告。
      * 實體感知：基於命名實體識別結果切塊。適用：簡歷、法律文件。
      * 主題切塊：使用主題模型將同一主題句子聚合。適用：新聞、多主題論文。
    * **適應模型與混合型**切塊：
      * Token級切塊：嚴格按模型可處理的 token 數量切割，確保不超限。適用： GPT 等 Transformer 模型。
      * 混合切塊：結合多種策略。優點：強大且靈活。缺點：實現最複雜。適用：極複雜的文件。


#### 2.2 高品質數據提取與處理工具

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

#### 2.3 選擇合適的嵌入模型 (Embedding)

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

### 資料檢索 (Data Retrieval)

#### 2.4 檢索策略：從單一到混合

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

### 檢索後處理 (Post-Retrieval Processing)

#### 2.5 Rerank：從「找得全」到「選得準」的關鍵一步；[更多 Embedding和Rerank模型說明在這](#Appendix-Embedding-Reranking-RAG)

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


### LLM 生成優化 (LLM Generation)

在 `xinference` 或 `Ollama` 中，不僅檢索與重排模型重要，最終用於生成答案的模型也應根據需求選擇。

如果你的 RAG 流程（從數據處理到 Rerank）已經做得非常好，檢索到的上下文品質極高，那麼有時並不需要動用最強大的生成模型（如 GPT-4 等級）。`Llama-3.1-70B-Instruct`就足以生成優質、準確的答案。這同樣是在**準確性和計算成本**之間做出明智的權衡。

### 迭代優化與評估 (Iterative Optimization & Evaluation)

建立 RAG 系統並非一勞永逸，它是一個需要持續優化和迭代的過程。當系統上線後，我們需要一套機制來衡量其表現。

業界常見的做法包括：

* **自動化評估：** 使用強大的 LLM（如 GPT-4o）作為「評審」，根據相關性、忠實度、簡潔性等指標，自動評估 RAG 系統的回應品質。
* **評估框架：** 利用 `Ragas` 等開源框架，對 RAG 管道的每個環節（檢索、生成）進行量化評分。
* **使用者回饋：** 建立簡單直觀的回饋機制（如點讚/點踩），收集使用者最直接的體驗，這是發現問題、改進系統最寶貴的資訊來源。

---

## 總結

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

> 📖 如需進一步了解，請參閱原文：  
> [https://blog.twman.org/2024/07/RAG.html](https://blog.twman.org/2024/07/RAG.html)

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "從零到一：打造本地端高精準度 RAG 系統的實戰指南",
  "description": "2025 最新 RAG 技術實戰指南，涵蓋環境部署、數據處理、混合檢索與 Rerank 優化技巧。",
  "keywords": "RAG, Retrieval-Augmented Generation, LLM, 檢索增強生成",
  "datePublished": "2024-07-07",
  "dateModified": "2026-01-02",
  "author": {
    "@type": "Person",
    "name": "TonTon Huang Ph.D.",
    "url": "https://twman.org/"
  },
}
</script>