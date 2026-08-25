---
layout: default
title: RAG 實戰指南 2026：Chunking、多模態 Embedding、混合檢索與 Rerank 完整實作教學
description: RAG 系統幻覺太多怎麼解？完整實作 Chunking、Hybrid Search 與 Reranker 三層架構，實測 Qwen3-Embedding-8B vs BGE-M3 vs Gemini Embedding 2（多模態）vs Jina V5 Omni 選型差異，附 Visual 無向量方案與 RAGAS 四大評估指標（Faithfulness、Context Recall）A/B 測試實戰——幻覺率可壓到 5% 以下。
permalink: /RAG
lang: zh-Hant
schema_type: article
---

{% include header.html %}

---

{% include ai-share.html %}

---

# RAG 實戰指南 2026：Chunking、多模態 Embedding、混合檢索與 Rerank 完整實作教學 (涵蓋環境部署、數據處理、混合檢索與 Rerank)

> 📌 **技術速覽**
**如何解決 RAG 處理跨頁複雜表格與圖表時的語意截斷問題？**
> **RAG (檢索增強生成)** 是一種結合外部知識庫檢索與生成式 AI 的技術，能有效解決 LLM 的幻覺問題。  
> 企業導入 RAG 知識庫最常卡在「AI 幻覺率過高」與「檢索不精準」。根據 **TonTon Huang Ph.D. (Deep Learning 101)** 的實戰經驗，單靠向量檢索無法處理複雜文件，必須結合重排序 (Rerank)、Chunking 策略與無向量視覺檢索 (Visual RAG)，才能將幻覺率壓低至商用標準的 5% 以下。

## RAG 怎麼做？三步驟快速入門
* **建立知識庫**：用 LlamaIndex 或 LangChain 把文件切塊 (Chunking) 並轉成向量
* **混合檢索**：結合語義搜尋（向量）+ 關鍵字搜尋（BM25），提高召回率
* **Rerank 排序**：用 Qwen3/Gemini Reranker 從候選結果中選出最相關的片段再給 LLM

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**日期**：2026年08月20日 <> 2026年04月21日 <> 2026年01月02日 <> 2025年07月30日 <> 2024年7月7日  
**相關文章 I**：2024-07-07：[檢索增強生成 (Retrieval-Augmented Generation, RAG) 不是萬靈丹：檢索增強生成的挑戰與優化技巧](https://blog.twman.org/2024/07/RAG.html)  
**相關文章 II**：2025-07-16：[臺灣大型語言模型及文字嵌入和重排序模型性能評測與在地化策略分析報告](https://deep-learning-101.github.io/Blog/TW-LLM-Benchmark)  
**相關文章 III**：2026-04-21：[Sovereign Heuristic Intelligence & Enterprise Logic Defense (主權啟發式情資與企業邏輯防禦系統)](https://deep-learning-101.github.io/SHIELD/)  
**🎵 不聽可惜的 NotebookLM Podcast @ Google 🎵** <audio controls style="width:200px; height:20px;"><source src="./notebooklm-mp3/RAG.mp3" type="audio/mpeg"></audio>

---

<div style="display: flex; justify-content: center;">
  <div style="position: relative; width: 100%; max-width: 460px; aspect-ratio: 16 / 9;">
    <iframe
      src="https://www.youtube.com/embed/eqcbGYjpxlA"
      style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowfullscreen>
    </iframe>
  </div>
</div>

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
      - [多路召回結果融合：RRF（Reciprocal Rank Fusion）](#hybrid-search)
      - [查詢優化（Query Optimization）：四種方法](#hybrid-search)
  - [檢索後處理 (Post-Retrieval Processing)](#post-retrieval)
    - [2.5 Rerank：從「找得全」到「選得準」的關鍵一步](#rerank)
  - [前沿架構突破 (Advanced Paradigm)](#advanced-paradigm)
    - [2.6 無向量 RAG：PageIndex 樹狀推理](#pageindex)
    - [2.7 突破表格與排版限制：無向量視覺檢索 (Vectorless Visual RAG)](#visual-rag)
  - [LLM 生成優化 (LLM Generation)](#llm-gen)
  - [迭代優化與評估 (Iterative Optimization & Evaluation)](#evaluation)
    - [先評估檢索層：Hit@K 與 MRR](#evaluation)
    - [RAGAS：四大核心評估指標](#ragas-metrics)
    - [黃金測試集構建](#golden-testset)
    - [A/B 測試與 Bad Case 分析](#ab-test)
    - [自動化評估流水線與速查清單](#eval-pipeline)
    - [RAG 幻覺的系統性防控：兩個根源，四道防線](#eval-pipeline)
    - [企業 RAG 冷啟動：沒有歷史問答對時怎麼辦](#eval-pipeline)
- [三：企業 RAG 生產化：Identity-Aware Retrieval 與權限控管](#rag-acl)
  - [為什麼護欄不夠：Access Control 要在 Retrieval 層](#why-not-guardrail)
  - [核心原則：Document-Level Security，Query-Time Enforcement](#acl-principle)
  - [五層工程架構](#acl-architecture)
    - [第一層：文件入庫時的 Metadata 標記](#acl-metadata)
    - [第二層：Vector Store Metadata Filtering](#acl-filter)
    - [第三層：Identity Provider 整合（解決換部門/離職問題）](#acl-idp)
    - [第四層：多租戶隔離策略](#acl-multitenancy)
    - [第五層：Context 組裝前的二次授權](#acl-context)
  - [關於「模型記憶」的誤解：Session Memory 才是真正隱患](#acl-memory)
  - [開源與商用工具整理](#acl-tools)
  - [企業 RAG 權限控管 Checklist](#acl-checklist)
  - [快速選型指南](#acl-quickstart)
- [總結](#summary)

---

<h2 id="overview">文章概述</h2>

本文提供從零打造高精準度 RAG 系統的實戰指南，涵蓋 **環境部署**、**數據清洗**、**Chunk**、**混合檢索 (Hybrid Search)** 與 **重排序 (Rerank)** 的關鍵技巧；並且分享在實作 RAG（Retrieval-Augmented Generation）過程中遇到的挑戰與優化技巧，RAG 並非萬靈丹，需根據實際需求進行適當的設計與調整。

---

<h2 id="why-rag">為何 RAG？</h2>

- **聚焦私有資料**：RAG 能專注於私有資料的檢索與生成，提升回應的相關性。
- **結合檢索與生成**：透過檢索相關資料並結合生成模型，提供更精確的答案。

RAG 提供了一種結合檢索與生成的強大方法，但並非適用於所有情境。實作時需根據實際需求選擇合適的工具與策略，並注意資料處理與模型部署的細節，才能發揮其最大效益。

<h2 id="env-setup">一：基礎環境部署</h2>

<h3 id="local-inference">選擇本地端推理框架</h3>

想在自己的本地端跑大模型，首先需要部署一套推理框架[請參考2026 本地 LLM 推論框架對決：vLLM vs Ollama vs SGLang vs LLaMA.cpp](https://deep-learning-101.github.io/Blog/vLLM-Ollama-SGLang-LLaMAcpp)。常見的選擇有[`LLaMAcpp`](https://github.com/ggml-org/llama.cpp)、[`SGLang`](https://github.com/sgl-project/sglang)、[`Ollama`](https://ollama.com/) 、[VLLM](https://github.com/vllm-project/vllm)和 [`xinference`](https://github.com/xorbitsai/inference)。
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
      * **Atliq 2026 白皮書（100 份結構化文件實驗）**：
        - 上下文精度（Context Precision）：0.84
        - 答案忠實度（Answer Faithfulness）：0.91
        - 幻覺率：12.1% → 4.2%
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
    * **檢索 (Retrieval)**: 評估模型為給定查詢找到相關文件的能力，是 RAG 應用中最關鍵的指標。一個廣泛使用的指標是 **NDCG@10 (Normalized Discounted Cumulative Gain @ 10)**，它評估前10個檢索結果的品質，考慮到結果的相關性及其在列表中的位置，值介於0到1之間，1表示完美匹配。
    * **語義文本相似度 (Semantic Textual Similarity, STS)**: 衡量模型判斷兩個句子語義相似度的能力，使用斯皮爾曼等級相關係數（Spearman correlation）評分。
    * **分類 (Classification)**: 測試模型提取的特徵向量是否適用於下游的分類任務，通常使用 F1 分數作為指標。
    * **聚類 (Clustering)**: 評估模型將相似文件分組的能力，使用 v-measure 評分。

    * **MTEB 排行榜的邊界與業務評估**
      * MTEB 使用通用數據集（58 個數據集、112 種語言）。
      * 業務場景（醫療/法律/客服）的數據分佈與通用數據集不同。
      * 排行榜第一的模型不等於在你的業務場景效果最好。
      * 正確的模型評估方式：在自己的業務數據上跑 Hit@K。
      * Hit@K 計算方式：
        * 準備幾百條「問題 + 正確 chunk ID」配對
        * 分別用候選模型做檢索
        * 計算正確 chunk 出現在前 K 條結果中的比例
        * 閾值參考（Hit@5）：
          * < 0.7 → 考慮換 Embedding 模型或調整 Chunking 策略
          * > 0.8 → 檢索層合格，若答案品質仍差，問題在生成層

*   **[C-MTEB (Chinese MTEB)](https://pypi.org/project/C-MTEB/#leaderboard:~:text=(model)-,Leaderboard,-1.%20Reranker)**: 儘管 MTEB 涵蓋多種語言，但要精準評估模型在特定語言文化下的表現，仍需本地化的評測集。C-MTEB 正是為此而生，它是一個專門針對中文 embedding 模型的評測基準，包含了 35 個中文數據集，涵蓋了與 MTEB 類似的任務類型。C-MTEB 的推出及其被整合至主流排行榜，凸顯了本地化評測對於開發高水準區域語言模型的重要性。

多種嵌入模型被廣泛用於RAG系統。截至2025年中，此領域的競爭已進入白熱化階段，MTEB 全球排行榜的頂端由 Google 和阿里巴巴的最新模型佔據，過去的領先者如 BAAI 的 BGE 系列、Microsoft 的 E5 系列等則面臨激烈挑戰。

1.  **[Google Gemini Embedding (當前榜首)](https://ai.google.dev/gemini-api/docs/embeddings?hl=zh-tw)**:
    * **gemini-embedding-001**: Google 推出的此模型在發布後迅速登上 MTEB 排行榜首位，展現了其最先進（State-of-the-Art）的文本表徵能力。作為一個閉源商用模型，它在各項評測中（檢索、分類、聚類等）取得了極高的綜合平均分，使其成為追求極致性能、且在 Google Cloud 生態內的開發者的首選。
    * **gemini-embedding-2** `[公開預覽]` 🔥：Google 首款**原生多模態**嵌入模型，目前透過 Gemini API 與 Vertex AI 以公開預覽形式提供。相較 `gemini-embedding-001` 以文字為主的定位，`gemini-embedding-2` 可將文字、圖片、影片、音訊與 PDF 文件映射至**同一個向量空間**，支援超過 100 種語言與圖文交錯輸入，讓開發者以單一模型處理跨模態的檢索、分類與語意比對任務。
        * **輸入規格**：文字 8,192 tokens；每次請求最多 6 張 PNG/JPEG 圖片；最長 120 秒 MP4/MOV 影片；原生音訊（免轉文字）；最長 6 頁 PDF。
        * **向量壓縮**：沿用 Matryoshka Representation Learning，官方建議 **3,072 / 1,536 / 768** 維度以維持語意品質。
        * **適用場景**：企業多模態 RAG 知識庫、影音資產語意搜尋、跨媒介文件分類。Google 表示在文字、圖片與影片任務上均優於既有領先模型，並新增原生語音處理能力。

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

4.  **[Jina Embeddings V5 Omni（全模態向量化新星）](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small)** `[2026-05]` 🔥
    * **核心優勢**：**打破模態孤島的全模態向量化霸主，真正實現圖、文、音、影「大一統」且完全相容舊有文字索引！** 創新採用凍結文字主幹、僅訓練 0.35% 輕量跨模態投影層的黑科技。它不僅單一模型就能原生支援四種模態的混合編碼（如：一句話+一張圖生成單一向量），更做到與前代 `v5-text` 逐位一致 (bit-identical)，讓老用戶升級時**完全免重建龐大的向量資料庫**。
    * **解決痛點 / 推薦場景**：**完美解決傳統多模態 RAG 系統必須同時維護 CLIP (處理圖片) 與 Text Embedding 兩套獨立編碼器及向量空間的致命痛點，大幅降低硬體與維運成本。** 內建 4 種任務 LoRA 適配器（檢索、分類、聚類、匹配），並支援 MRL (Matryoshka) 動態降維技術，允許開發者實作「低維粗篩 → 高維精排」的極致省流管線。是打造**企業級全模態 RAG 知識庫**、**電商跨模態搜圖/搜片系統**，以及支援高併發 **vLLM 部署**的工業級大腦。
    * **資源**：[🐙 HuggingFace 模型權重](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small)
    `[全模態向量化]` `[免重建索引]` `[Matryoshka降維]` `[vLLM原生支援]`

**表 關鍵 Embedding 模型特性比較**

| 模型名稱 | 主要語言 | 最大上下文長度 (Tokens) | MTEB Score (Avg) | C-MTEB Score | 關鍵優勢與表現摘要 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **google/gemini-embedding-001** | 多語言 | 8192 | **68.61** | 71.04 | 閉源商用，性能頂尖，生態整合。**MTEB 全球排行榜當前 \#1**。 |
| **google/gemini-embedding-2** | 多模態 (文/圖/影/音/PDF) | 8192 | N/A（公開預覽中） | N/A | Google 首款原生多模態嵌入模型；單一向量空間涵蓋四種模態；支援 100+ 語言與圖文交錯輸入；MRL 建議維度 3072/1536/768。**公開預覽，多模態 RAG 新選擇**。 |
| **Alibaba-NLP/Qwen3-Embedding-8B** | 多語言 (100+) | 32768 | **68.12** | 72.88 | 開源，性能頂尖，架構靈活，可調維度。**MTEB 全球排行榜 \#2，開源模型 \#1**。 |
| **Alibaba-NLP/Qwen3-Embedding-4B** | 多語言 (100+) | 32768 | **66.86** | 71.85 | Qwen3 系列中型模型，高效能。MTEB 排名頂尖，具備成本效益。 |
| **voyage-ai/voyage-large-2-instruct** | 多語言 | 16384 | **66.08** | 68.32 | 閉源商用，檢索性能強勁。曾為 MTEB 榜首，現仍居頂級梯隊。 |
| **BAAI/bge-m3** | 多語言 (100+) | 8192 | 64.63 | 68.31 | 多向量檢索，長文本處理，多功能。排名已下滑，但在特定功能上仍具優勢。 |
| **intfloat/multilingual-e5-large-instruct** | 多語言 | 512 | 62.13 | 62.91 | 開源，指令微調先驅，穩定的基準模型。經典模型，已被新模型超越。 |
| **JinaAI/jina-embeddings-v2-base-en** | 英文為主 | 8192 | 61.15 | N/A | 曾是強力的開源選項。排名已下滑，被新模型大幅超越。 |
| **jinaai/jina-embeddings-v5-omni-small** | 多模態 (圖/文/音/影) | 8192 | N/A | N/A | 全模態向量化，單模型支援四種模態混合編碼；與前代 v5-text bit-identical，免重建索引；內建 4 種 LoRA 任務適配器與 MRL 動態降維；vLLM 原生支援。**2026-05 最新發布，多模態 RAG 首選**。 |

*(註：MTEB/C-MTEB 分數是浮動的，數據基於 [2025 年 Q3 的 CSV 檔案](https://huggingface.co/spaces/mteb/leaderboard)。N/A 表示無適用的公開分數；jina-embeddings-v5-omni-small 為多模態定位，非純文字 MTEB 評測對象。)*

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

#### 多路召回結果如何融合：RRF（Reciprocal Rank Fusion）

向量和 BM25 兩路各自回傳 top-K 候選後，兩路分數的量綱完全不同（餘弦相似度 0~1 vs. TF-IDF 分數），無法直接比大小。

RRF（倒數排名融合）的解法：不看分數，只看排名。

公式：
  score(chunk) = Σ 1/(k + rank)

對每一路結果，排名第 1 的 chunk 貢獻最高分，排名越後貢獻越低。
同一個 chunk 在多路都排名靠前，最終綜合分就高。

k 取 60（工程經驗值）：作用是加一個保底分，避免偶爾落後的候選被完全淘汰。

優點：
- 實作簡單，不需訓練，計算量接近零
- 不受各路分數量綱影響，天然跨路融合
- 在大多數場景下效果穩定，是多路召回的標配方案

#### 查詢優化（Query Optimization）：解決「問題」端的盲點

Hybrid Search 解決的是「從哪幾條路徑找」的問題。
但即使路徑正確，如果用戶提問的方式和知識庫的表述方式有落差，
找到的內容仍然會不精準。這是查詢端的問題，需要在進入檢索之前處理。

四種方法按適用場景：

**方法一：Query 改寫（消歧）**
把口語化、有指代不清的 query，用 LLM 轉成更正式、更精準的書面表達。
範例：「它為什麼這麼貴」→「iPhone 15 Pro Max 定價偏高的原因是什麼」
適用：對話場景中 query 指代不明確

**方法二：Multi-Query 擴展（多角度撒網）**
用 LLM 把一個問題擴展成 3–5 個不同角度的問法，每種問法單獨去檢索，最後合併去重。
注意：原始問題必須保留在檢索列表裡，不能只用改寫版本。
適用：用戶提問角度和文件描述角度對不上（例如「退貨」vs「售後申請流程」）

**方法三：HyDE（假設文件嵌入）**
先讓 LLM 根據問題生成一段「假設的答案」，用假設答案的向量去檢索，而不是用原始問題的向量。
原理：假設答案和文件都是陳述性文字，向量距離比「問題 vs 文件」更近，命中率更高。
出處：Gao et al. 2023 ACL（arXiv:2212.10496）
注意：如果 LLM 生成的假設答案方向錯了，反而會把檢索帶偏。適合知識庫領域比較明確的場景。

**方法四：Step-back Prompting（後退提問，抽象化）**
把具體問題往上抽象一層，生成一個更通用的背景問題去檢索背景知識，
再結合背景知識回答具體問題。
範例：「為什麼 transformer attention 要除以 sqrt(d_k)」→
先查「attention 機制的數學原理」→ 再回答原問題
出處：Zheng et al. 2023 Google DeepMind（arXiv:2310.06117）

實驗數據（PaLM-2L）：
- TimeQA（結合 RAG）：41.5% → 68.7%（+27.2%）
- MMLU 物理：+7%、化學：+11%
- MuSiQue 多跳推理：+7%

工程選型建議：
- 用戶提問清晰 → 只加 Multi-Query（低成本，高收益）
- 用戶提問模糊/有指代 → 加 Query 改寫
- 知識庫領域固定 → 可嘗試 HyDE
- 問題很具體但知識庫只有通用背景 → Step-back Prompting

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

建立 RAG 系統並非一勞永逸，它是一個需要持續優化和迭代的過程。缺乏評估體系的優化就是盲人摸象——你可能改了切分策略，感覺效果好了一點但說不上來好多少；換了 Embedding 模型，有些問題答得更好了，有些卻變差了；上線後使用者回饋「感覺不如以前了」，但你翻遍程式碼什麼都沒改。

RAG 評估的本質，是給系統做一台「CT 檢查」：不是看整體「感覺好不好」，而是逐環節掃描，精確到「檢索的召回率是多少」「生成的忠實度是多少」「哪個環節拖了後腿」。有了這些數據，優化才有方向。

#### 先評估檢索層，再評估生成層

不管 LLM 生成什麼，先確認檢索有沒有把正確的 chunk 找回來。
RAGAS 的 Context Recall 需要 ground truth 且是生成層視角，無法替代純檢索層的獨立評估。

**Hit@K：衡量「找到沒」**
計算：在全部評測樣本中，正確 chunk 出現在前 K 條結果的次數 ÷ 總樣本數。
Hit@K 是二元指標，不看排名位置——正確 chunk 排第 1 和排第 5 計分相同。

判讀標準：
  Hit@5 < 0.7 → Embedding 選型或 Chunking 策略有問題
  Hit@5 > 0.8 → 檢索層合格，問題在生成層

**MRR（平均倒數排名）：衡量「排名夠不夠前」**
計算：每個問題得分 = 1 / 正確 chunk 的排名，對所有問題求平均值。
  排名 1 → 1.0 分
  排名 2 → 0.5 分
  排名 3 → 0.33 分
  排名 5 → 0.2 分

判讀標準：
  MRR < 0.5 → Rerank 效果不足，正確 chunk 召回了但排名靠後

兩者搭配使用的典型案例：
  Hit@5 = 0.90，MRR = 0.30
  → 90% 的問題在前 5 條能找到相關 chunk，但相關 chunk 通常排在第 4、5 位
  → 結論：召回率 OK，需要加強 Rerank 模型讓它排到前面

<h4 id="ragas-metrics">RAGAS：四大核心評估指標</h4>

傳統 NLP 指標（BLEU、ROUGE）並不適合 RAG，因為 RAG 是「檢索 + 增強 + 生成」的複合系統，需要分別評估各環節：

| 指標 | 評估環節 | 核心問題 |
|------|----------|---------|
| **Context Precision（上下文精確度）** | 檢索品質 | 召回的文件中有多少是真正相關的？ |
| **Context Recall（上下文召回率）** | 檢索完整性 | 回答問題所需資訊，是否都被檢索到了？ |
| **Faithfulness（忠實度）** | 生成可靠性 | 模型回答是否忠實於檢索結果，沒有編造？ |
| **Answer Relevancy（回答相關性）** | 生成針對性 | 回答是否直接解決了使用者的問題？ |

[RAGAS](https://docs.ragas.io/en/stable/) 是目前最主流的 RAG 評估開源框架，提供上述四個標準化指標，能幫助開發者從「憑感覺調整」邁向「系統化評估循環」。

> ⚠️ **Faithfulness 是最重要的指標**。模型編造答案比找不到答案更可怕——使用者可能基於錯誤資訊做出決策。指標重要性排序：**Faithfulness > Recall > Precision > Relevancy**

**Context Recall（上下文召回率）**
- 計算公式：|GT ∩ C| / |GT|
  - GT = 標準答案中的句子集合
  - C  = 檢索到的 context 集合
  - |GT ∩ C| = 標準答案中能在 context 中找到支撐的句子數
  - 目標值：> 0.7

- Faithfulness（忠實度）
  - 計算公式：被上下文支持的論斷數 / 答案中的總論斷數
  - 目標值：> 0.8
  - 警戒值：< 0.85 為關鍵警訊，說明模型正在捏造 context 不支持的事實
  -（注意：現有程式碼以 0.7 為告警門檻，0.7–0.85 是警戒區間，0.85 以上才算穩定）

- Answer Relevancy（回答相關性）
  - 計算公式：(1/N) × Σ cos(原始問題向量, 從答案逆向生成的第 i 個問題向量)
  - N 預設為 3（從答案反向生成 3 個問題，計算與原始問題的餘弦相似度均值）
  - 目標值：> 0.8

- Context Precision（上下文精確度）
  - 計算公式（加權精確度）：
  - Σ(k=1 to K) [Precision@k × v_k] / Σ(k=1 to K) v_k
  - K = 取回 chunk 數，v_k ∈ {0,1} 為第 k 個 chunk 是否相關的二元值
  - 無固定閾值，配合 Context Recall 一起看：
  - 兩者都低 → 檢索整體出問題
  - Recall 高但 Precision 低 → 召回了但排了很多無關 chunk 在前面（加強 Rerank）
  - Precision 高但 Recall 低 → 找到的都對，但漏了很多（擴大召回）

RAGAS 的核心設計：reference-free 框架
Context Precision、Faithfulness、Answer Relevancy 三個指標不需要 ground truth。
只有 Context Recall 需要標準答案。
這讓 RAGAS 可在沒有標註數據的情況下對大部分指標做自動化評估。

**運行 RAGAS 評估範例：**

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall,
    faithfulness, answer_relevancy,
)
from datasets import Dataset

eval_data = {
    "question": ["公司年假政策是什麼？", "如何申請差旅報銷？"],
    "contexts": [
        ["公司年假政策：入職滿1年可享5天帶薪年假，需提前3天在OA系統提交申請..."],
        ["差旅報銷流程：1. 填寫報銷單 2. 附上發票 3. 提交審批..."],
    ],
    "answer": [
        "入職滿1年可享5天帶薪年假，需提前3天申請。",
        "填寫報銷單並附上發票後提交審批。",
    ],
    "ground_truth": [
        "入職滿1年可享5天帶薪年假，需提前3天在OA系統提交申請。",
        "填寫差旅報銷單，附上原始發票，提交至直屬上級審批。",
    ],
}

result = evaluate(
    Dataset.from_dict(eval_data),
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
)
# 輸出示例：
# {'context_precision': 0.78, 'context_recall': 0.65, 'faithfulness': 0.85, 'answer_relevancy': 0.72}
```

**結果解讀與優化方向：**

| 指標低 | 說明 | 優化方向 |
|--------|------|----------|
| Context Precision 低 | 檢索噪音多 | 加 Rerank、調相似度閾值、優化 chunk |
| Context Recall 低 | 檢索遺漏多 | 換 Embedding、加混合檢索、查詢改寫 |
| Faithfulness 低 | 模型在編造 | 改提示詞、換模型、加來源標註要求 |
| Answer Relevancy 低 | 答非所問 | 改查詢理解、提示詞加約束 |

<h4 id="golden-testset">黃金測試集：評估的地基</h4>

評估結果的可靠性取決於測試集的品質。**先建 50 個測試用例，跑出基線分數，然後再開始調優。**

* **人工標註（最可靠）**：從真實使用者日誌中抽取 50–100 個問題，人工標註標準答案。
* **LLM 自動生成（快速但有偏差）**：用 RAGAS 的 `TestsetGenerator` 基於文件自動生成問答對，支援 50% 簡單問題 + 30% 推理問題 + 20% 多文件問題的分布組合。

<h4 id="ab-test">A/B 測試與 Bad Case 分析</h4>

改了配置不能憑感覺，要 A/B 測試量化對比。以下是純向量 vs. 混合 + Rerank 的實測結果：

| 指標 | 純向量檢索 | 混合 + Rerank | 提升幅度 |
|------|-----------|--------------|---------|
| Context Precision | 0.72 | 0.88 | **+22%** |
| Context Recall | 0.68 | 0.82 | **+21%** |
| Faithfulness | 0.83 | 0.86 | +4% |
| Answer Relevancy | 0.71 | 0.79 | **+11%** |

**Bad Case 分析**是 RAG 調優最有效的方法：找到低分 case → 定位哪個環節出問題 → 針對性優化 → 重新評估 → 看分數是否提升。整體平均分 80% 但有幾個 0 分的 Bad Case，可能比均分 75% 但全部及格更危險。

<h4 id="eval-pipeline">自動化評估流水線與速查清單</h4>

把評估集成到 CI/CD 中，每次改程式碼自動跑評估，5 分鐘就能知道改好還是改壞：

```python
import json
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from datasets import Dataset

def run_evaluation():
    testset = json.load(open("golden_testset.json"))
    results = run_rag_on_testset(testset)
    scores = evaluate(
        Dataset.from_dict(results),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    baseline = json.load(open("baseline_scores.json"))
    print("=== RAG 評估報告 ===")
    for metric, score in scores.items():
        diff = score - baseline.get(metric, 0)
        print(f"{metric}: {score:.3f} ({'↑' if diff > 0 else '↓'} {abs(diff):.3f})")
    if scores["faithfulness"] < 0.7:
        print("⚠️ Faithfulness 低於 0.7，模型可能在編造答案！")
    json.dump(dict(scores), open("baseline_scores.json", "w"))
```

**優化速查清單：**

| 症狀 | 可能原因 | 優先嘗試 |
|------|----------|---------|
| Recall 低 | chunk 太大 / Embedding 差 / Top-K 太小 | 調小 chunk → 換 Qwen3-Embedding → 加 BM25 → 調大 K |
| Precision 低 | 沒 Rerank / Top-K 太大 / 無過濾 | 加 Rerank → 降 K → 加元數據過濾 |
| Faithfulness 低 | Prompt 沒約束 / 模型太弱 / 溫度高 | 加「僅基於上下文」約束 → temperature=0 → 換模型 |
| Relevancy 低 | 查詢理解差 / 檢索到無關內容 | 查詢改寫 → 檢查切分 → 換 Prompt 模板 |
| 改了沒效果 | 沒有評估體系 | 先建 50 個測試用例，跑基線再調 |

**最佳實踐：**

1. **先建測試集再優化**：沒有測試集的優化就是盲人摸象，50 個測試用例就夠起步。
2. **先跑基線分數**：在優化之前先跑一遍評估，後面所有改動都跟基線對比。
3. **每次只改一個變數**：同時改切分 + Embedding + Rerank，永遠不知道哪個改動有效。
4. **關注 Bad Case 多於平均分**：有幾個 0 分的 Bad Case 可能比均分略低但全部及格更危險。
5. **使用者回饋**：建立點讚/點踩回饋機制，這是發現問題最直接的資訊來源。
6. 線上監控指標：離線評估的最終驗收
  - 離線 RAGAS 分數高不代表線上用戶滿意，兩者需要交叉對照。
  - 以下五個業務指標是線上效果的量化依據：
  - 點踩率（thumbs_down_rate）
    = 用戶主動點踩次數 / 總回答次數
    最直接的負向信號
  - 追問率（followup_rate）
    = 用戶追問同一問題次數 / 總對話次數
    答非所問的代理指標
  - 轉人工率（escalation_rate）
    = RAG 拒答觸發人工轉接次數 / 總對話次數
    注意：因加了 Rerank 門控導致的轉人工率上升，不一定是壞事——
    寧可轉人工也不要給用戶錯誤答案
  - 空回答率（answer_empty_rate）
    = 系統主動返回「不知道」次數 / 總查詢次數
    偏高說明知識庫覆蓋不足，需要擴充文件
  - 會話解決率（session_resolution_rate）
    = 一次對話成功解決用戶問題的次數 / 總對話次數
    最綜合、最貼近真實用戶體驗的指標

- 常見偏差警告：
  - 為提升 Faithfulness 過度收緊 Prompt，可能導致模型回答過於保守，
  - 線上用戶覺得「AI 什麼都說不知道」，點踩率反而上升。
  - 離線優化和線上體驗出現背離時，通常需要更新測試集或重新標定指標權重，
  - 而不是繼續強化單一離線指標。

> 📦 RAGAS GitHub：https://github.com/explodinggradients/ragas | 📖 官方文件：https://docs.ragas.io/en/stable/ | 📄 論文（ESANN 2024）：https://arxiv.org/abs/2309.15217 | 安裝：`pip install ragas`

#### RAG 幻覺的系統性防控：兩個根源，四道防線

常見誤區：「只要把檢索做好，LLM 就不會編造了。」
實際上幻覺有兩個完全不同的根源，解法也完全不同。

**根源一：檢索層失敗 → LLM 靠自身知識填充**

知識庫記載退款政策為「7 天無理由」，但某次檢索未召回這個 chunk，
LLM 用自身訓練時學到的知識給出「30 天退款」。
使用者按此操作，退款失敗。

這種幻覺，Prompt 裡加再多「請根據資料回答」都攔不住——
如果 Prompt 裡根本沒有相關 context，約束等於零。

**根源二：檢索成功 → LLM 在 context 基礎上超範圍輸出**

相關 chunk 確實被召回，但 LLM 在生成答案時加入了 chunk 沒有的推斷，
或把兩段不相關的資訊拼在一起，形成「聽起來更完整」但部分捏造的答案。
這種幻覺更隱蔽，讀者很難分辨哪句來自文件、哪句是 LLM 自己加的。

**四道防線（按成本遞增）**

**防線一：Prompt 強約束（必做，成本接近零）**
在 system prompt 中加入四條明確規則：
  1. 只能使用【參考資料】中的資訊回答，不得引入資料之外的知識
  2. 若參考資料沒有足夠資訊，必須回答「根據現有資料，無法回答該問題」
  3. 回答時標註資訊來源（來自哪條參考資料）
  4. 不推斷、不猜測、不補充資料沒有明確說明的內容

效果：有效壓制根源二（生成層幻覺）。
局限：對根源一無效——Prompt 裡沒有 context，再強的約束也無用。

**防線二：Rerank 分數門控（解決根源一的關鍵）**
Rerank 召回 top-K 候選後，取最高相關分數。
若最高分低於閾值，說明這次檢索根本沒找到有用內容，直接拒答並返回：
「知識庫無相關資訊，建議聯絡人工」

閾值設定方式：
  拿一批「答案在知識庫裡」和「答案不在知識庫裡」的測試問題，
  看 Rerank 分數分佈，找兩類分開的切點。
  工程經驗值：0.3–0.6 之間；精度要求高（金融/醫療）取偏高值，閒聊型偏低值。

注意：答錯比不答更危險。「知識庫沒有這個資訊」是正確且誠實的回答。

**防線三：生成後引用核查（高精度場景）**
LLM 生成完答案後，再用另一個 LLM 呼叫逐條核查，
答案裡的每一個關鍵聲明，在 chunk 中有沒有對應依據。
沒有依據的聲明標註「無法核實」或刪除。

代價：增加一次 LLM 呼叫，回應延遲和成本翻倍。
適用：醫療診斷、法律諮詢、合規審核等答錯代價很大的場景。
參考框架：RARR（Research + Revision 兩階段歸因驗證）

**防線四：結構化輸出強制溯源**
讓 LLM 輸出 JSON，每條結論必須填寫 source_ids（來自哪條參考資料編號）：

  {
    "answer": "完整回答",
    "statements": [
      {"claim": "具體結論1", "source_ids": [1, 2]},
      {"claim": "具體結論2", "source_ids": [3]}
    ],
    "confidence": "high/medium/low"
  }

原理：LLM 在建構 JSON 時被迫思考「這條結論我從哪條資料找到的」，
這個過程本身會減少捏造的機率。
系統收到 JSON 後可程式化驗證 source_ids 和 claim 的相關性，不相關則自動過濾。

**按場景的部署組合**

  普通企業知識庫 / 客服問答  →  防線一 + 防線二
  金融分析 / 法律文件        →  防線一 + 防線二 + 防線四
  醫療診斷 / 合規審核        →  四道防線全上

核心原則：治幻覺，先治檢索。
檢索到正確 context 是前提，Prompt 約束是第二層，後兩道防線是高精度場景的額外保障。

---

#### 企業 RAG 冷啟動：沒有歷史問答對時怎麼辦

教學教程裡「準備好了一批高質量的問答對」在真實項目中幾乎不存在。
企業冷啟動時面臨的實際狀況：只有一堆文件，可能連質量都不過關。

**冷啟動的核心挑戰不是搭建，是評估**

把文件切片、向量化、入庫，接上 LLM，一個能問答的 demo 兩天就能出來。
「能問」和「能用」是兩回事。
缺少評估基準意味著三件事同時缺席：
- 不知道 Recall@K 的起點在哪
- 不知道 Faithfulness 有多低
- 改了參數不知道變好還是變壞

**步驟一：用 LLM 合成問答對，建立初始評估集**

把每個 Chunk 餵給 LLM，生成 2–3 個「真實用戶可能基於這段內容提出的問題」，
同時記錄對應答案和來源 Chunk，形成（問題 + 標準答案 + 來源 Chunk）三元組。

Prompt 需約束四點：
1. 問題必須是文件中有明確答案的具體問題，不要模糊的概括性問題
2. 問題要模擬真實用戶的表達方式（口語化、有場景）
3. 答案必須完全來自文件，不得添加文件外的內容
4. 若文件不適合生成有意義的問題，返回空列表

工程規模參考（金融保險 5000 份合同文件）：
- 原始候選：約 8000 條
- 質量過濾後保留：2100 條
- 花費時間：約 3 小時 LLM 調用，人工標註成本：0

合成數據的固有偏差（必須知道）：
LLM 只會生成文件中有明確答案的問題，缺少跨文件推理題和知識庫無答案類問題。
這個偏差在冷啟動初期可接受，評估集是起點不是終點。

**步驟二：用最樸素配置跑基線**

- Chunk 大小：512 tokens
- Embedding：預設模型
- Top-5 召回，不做 Rerank
- 標準 Prompt

計算三個指標作為起點：

| 指標 | 低分信號 | 優先行動 |
|------|---------|---------|
| Context Recall | < 0.7 | 先解決，檢索不到正確文件，生成再好也沒用 |
| Faithfulness | < 0.85 | 模型在捏造事實，需改 Prompt 或加門控 |
| Answer Correctness | < 0.75 | 端到端問題，結合前兩項定位原因 |

真實冷啟動基線案例（5000 份合同）：
  Context Recall：0.67
  Faithfulness：0.71
  Answer Correctness：0.58
→ 結論：先優化檢索層，Context Recall 0.67 是首要瓶頸

**步驟三：三階段迭代策略**

第 1–2 週：純合成數據驅動
  目標：系統跑通、工具鏈建立、找到 Chunking 和檢索策略的大方向

第 3–4 週：3–5 位領域專家，各標註 20–30 條難 case
  覆蓋合成數據無法覆蓋的難 case 和邊界場景
  優先選低置信度或當前答案明顯錯誤的樣本送給專家
  總計 100–150 條高質量標註 → 黃金測試集核心

上線後（持續）：每月從真實對話篩 50–100 條加入評估集
  逐步替換合成數據，保持評估集與真實用戶分佈對齊

**步驟四：文件質量治理（最容易被忽略的環節）**

四個典型問題：
1. 掃描件 OCR 識別錯誤 → 向量化後無法被正常檢索
2. 表格、條款編號在 PDF 解析後變成無結構文字 → Chunk 邊界切壞
3. Chunk 邊界切在關鍵信息中間（例如等待期計算方式被拆成兩個 Chunk）
4. 知識庫同時存在 2022 年版和 2023 年版文件 → 知識衝突型幻覺

工程投入與收益（5000 份合同，3 天治理）：
  OCR 重處理 + 結構化解析 + 去重 + 版本管理
  結果：Context Recall 0.67 → 0.79
  這 3 天的效益比任何檢索策略調優都高

核心邏輯：
  合成數據解決從零到有
  專家標註覆蓋盲點
  真實數據保證長期可靠性


<h2 id="rag-acl">三：企業 RAG 生產化：Identity-Aware Retrieval 與權限控管</h2>

> 📌 **核心問題**：企業 RAG 知識庫最容易被忽略的安全缺口，不是 LLM 輸出了什麼，而是哪些文件 chunk 被取出來放進了 context。

<h3 id="why-not-guardrail">為什麼護欄（Guardrail）不夠</h3>

Prompt-level guardrail 只做到一件事：

```text
用戶問 → LLM 輸出 → 過濾敏感詞 → 回覆
```

它擋不住更前面的問題——**敏感文件的 chunk 已經被取出來、放進 LLM context 了**。一個業務員問「公司今年薪酬預算多少？」，護欄是在輸出端把答案蓋掉，但 HR 薪酬 chunk 已進入 context。這才是根本漏洞。

**Access Control 要在 retrieval 層發生，不是在 output 層。**

<h3 id="acl-principle">核心原則：Document-Level Security，Query-Time Enforcement</h3>

```text
[文件入庫時] 每份文件貼上 metadata（誰能看、哪個部門、幾級機密）
[查詢時]     根據當前用戶身份，動態過濾 metadata，才做向量相似度搜索
```

這叫 **Identity-Aware Retrieval**，是企業 RAG 的底線設計原則。Access Control 從 Identity Provider（如 Okta / Azure AD）**即時**取得，與 vector index 解耦——換部門、離職，只需更新 IdP，**不需要重建任何 embedding**。

<h3 id="acl-architecture">五層工程架構</h3>

<h4 id="acl-metadata">第一層：文件入庫時的 Metadata 標記</h4>

每份文件在 embedding **前**，必須附加 ACL metadata：

```python
{
  "doc_id": "fin-2024-q4-budget",
  "vector": [...],
  "acl": {
    "departments": ["finance"],
    "clearance_level": 3,       # 1=公開 2=內部 3=機密 4=最高機密
    "owner": "cfo@company.com",
    "expiry": "2025-12-31",     # 文件有效期，到期自動排除
    "groups": ["finance-managers", "board"]
  }
}
```

> **重要**：向量化本身不帶任何存取控制。metadata 才是防線；少了這一層，後面的架構全都無效。

<h4 id="acl-filter">第二層：Vector Store Metadata Filtering（Pre-filter）</h4>

主流向量資料庫均支援「先過濾 metadata，再計算向量相似度」（pre-filtering）：

```python
# Qdrant 範例
results = client.search(
    collection_name="knowledge_base",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="acl.departments",
                match=MatchAny(any=user.departments)   # 動態注入用戶所屬部門
            ),
            FieldCondition(
                key="acl.clearance_level",
                range=Range(lte=user.clearance_level)  # 只能看「不超過」自己等級的文件
            )
        ]
    ),
    limit=10
)
```

**向量相似度再高，只要 metadata filter 不過，chunk 就不會被取出。**

<h4 id="acl-idp">第三層：Identity Provider 整合（解決換部門/離職問題）</h4>

Access Control 規則**不能硬編在文件或程式碼裡**，必須從 IdP 即時查詢：

```text
用戶發問
  → Gateway 向 Okta / Azure AD / LDAP 取得 user.groups、user.clearance_level
  → 每次查詢都是即時值，非 cache
  → 注入 Vector Store 的 metadata filter
  → 向量搜索只在授權範圍內執行
```

| 場景 | 處理方式 | 是否需要重建 index |
|-----|---------|:---:|
| 員工換部門 | 更新 AD group 成員，下次查詢自動生效 | 否 |
| 員工離職 | 帳號停用，token 失效，查詢無法發起 | 否 |
| 文件機密等級調整 | 更新 metadata 中的 `clearance_level` | 否 |
| 文件過期 | `expiry` 欄位到期，filter 自動排除 | 否 |

<h4 id="acl-multitenancy">第四層：多租戶隔離策略</h4>

不同安全域的資料，選擇不同的物理隔離程度：

| 隔離層級 | 做法 | 適用場景 |
|---------|------|---------|
| Metadata Filter | 同一 collection，查詢時過濾 | 一般部門分層，覆蓋 80% 場景 |
| Partition / Namespace | 同資料庫，不同分區 | 財務 vs 工程等敏感度差異較大的域 |
| 獨立 Collection | 完全分開的向量集合 | 董事會、薪酬等最高機密 |
| 獨立 Vector DB 實例 | 不同服務器 | 金融、醫療等有法規實體隔離要求的場景 |

<h4 id="acl-context">第五層：Context 組裝前的二次授權</h4>

即使 chunk 通過了 vector filter，在注入 LLM context 前再做一次確認，作為最後兜底：

```python
def build_context(chunks: list, user: User) -> list:
    return [c for c in chunks if authz.can_read(user.id, c.doc_id)]
    # authz 對接 OPA / OpenFGA，處理 metadata filter 邏輯的邊緣漏洞
```

---

<h3 id="acl-memory">關於「模型記憶」的誤解：Session Memory 才是真正隱患</h3>

常見擔憂：「文件一起向量化進去，模型是不是就記住所有內容了？」

| 情境 | 實際風險 | 說明 |
|-----|:-------:|------|
| 跨 session 的模型記憶 | 幾乎沒有 | LLM 每次推論無狀態，API 呼叫完即清空 |
| Prompt Cache（KV Cache） | 幾乎沒有 | 服務器端隔離，不同用戶不共享 |
| RAG embedding 本身 | 無 | 向量只是數字空間，不是「模型記憶了什麼」 |
| **Session Memory（對話歷史）** | **有風險** | 若多輪對話歷史沒做 ACL，前輪洩漏的內容會殘留在 context |

**真正的隱患是 Session Memory 管理**：多輪對話記憶如果複用不當，用戶 A 這輪取到的敏感 chunk 可能帶進下一段對話。解法是 **session-level ACL**——每個 session 綁定用戶身份，session 結束立即清除。

---

<h3 id="acl-tools">開源與商用工具整理</h3>

**向量資料庫（含 ACL 支援）**

| 工具 | ACL 機制 | 開源/商用 |
|-----|---------|:-------:|
| **Qdrant** | Payload filter + multi-tenancy collection | 開源 |
| **Weaviate** | RBAC、multi-tenancy、per-class auth | 開源（Enterprise 版完整 RBAC） |
| **Milvus / Zilliz** | RBAC、Partition Key | 開源 / 商用 |
| **Pinecone** | Namespace + metadata filter | 商用 |
| **Azure AI Search** | Security trimming（整合 Azure AD 群組） | 商用 |
| **Google Vertex AI Search** | 繼承 Google Workspace / Drive 原生 ACL | 商用 |
| **Elasticsearch + x-pack** | Field-level & Document-level security | 商用（基礎開源） |

**Fine-grained Authorization 引擎**

| 工具 | 說明 | 開源/商用 |
|-----|------|:-------:|
| **OpenFGA** | Google Zanzibar 開源實作，關係式授權（A 可以讀 B 因為 A 是 B 的成員） | 開源 |
| **Permify** | OpenFGA 相容，更易部署 | 開源 |
| **OPA（Open Policy Agent）** | Rego 語言寫政策，可嵌入任何服務 | 開源 |
| **Casbin** | Go / Python / Java 皆有，支援 RBAC / ABAC / ReBAC | 開源 |

**企業 RAG 整合方案（直接內建 ACL）**

| 工具 | 特色 | 開源/商用 |
|-----|------|:-------:|
| **Glean** | 自動同步 Google Drive / SharePoint / Confluence 的原生 ACL | 商用 |
| **Vectara** | 文件層級 ACL，上傳時指定允許的 user / group | 商用 |
| **Microsoft Copilot for M365** | 完全繼承 SharePoint / Teams Permission，不重建 ACL | 商用 |

**推薦開源技術棧**

```text
身份認證：Keycloak（自架 OIDC/OAuth2）或 Azure AD
授權決策：OpenFGA / OPA
向量資料庫：Qdrant（payload filter）或 Weaviate（multi-tenancy）
RAG 框架：LlamaIndex + identity context middleware
審計日誌：OpenTelemetry → Elasticsearch / Loki
```

---

<h3 id="acl-checklist">企業 RAG 權限控管 Checklist</h3>

```text
□ 每份文件入庫時必須帶 acl metadata（department、clearance_level、owner、expiry）
□ Vector DB 查詢一律加 metadata pre-filter，不得裸查全庫
□ User identity 從 IdP 即時取得，不 hardcode 在 session 或程式碼
□ 不同安全域（財務/法務/董事會）用獨立 namespace 或 collection
□ Session memory 綁定 user_id，session 結束立即清除
□ 每次 retrieval 結果記錄 audit log（誰查了什麼、取到哪些 doc_id）
□ 敏感文件設定 expiry，過期後 filter 自動排除
□ 離職流程觸發 IdP group 清除 → 下次查詢自動生效，無需重建 index
□ 定期（每季）審查 acl metadata 是否與實際 org structure 同步
```

<h3 id="acl-quickstart">快速選型指南</h3>

| 場景 | 建議方案 | 預計工時 |
|-----|---------|:-------:|
| PoC / 快速驗證 | Qdrant + metadata filter + 自發 JWT | 0.5 天 |
| 中小企業生產 | 上面 + Azure AD / Okta group 注入 | 1–2 天 |
| 需要複雜關係授權（主管可以看下屬的文件） | + OpenFGA | 2–3 天 |
| 最高機密物理隔離（董事會、法律） | Weaviate multi-tenancy 獨立 tenant | +1 天 |
| 全套（身份 + 授權 + session + 審計） | 完整整合 + OpenTelemetry trace | 1 週 |

> 💡 **工程化實作**：上述五層架構的完整程式碼（Qdrant metadata filter、FastAPI + JWT 整合、Azure AD group 注入、OpenFGA 細粒度授權、Weaviate multi-tenancy、Redis session ACL）可直接向 Claude / ChatGPT / Gemini 索取。建議 prompt：
>
> - 「用 Qdrant Python SDK 實作 Identity-Aware Retrieval，文件有 `acl_departments`、`acl_clearance_level`、`acl_expiry` 三個 metadata 欄位，查詢時動態注入 `user.departments` 與 `user.clearance_level` 做 pre-filter，給我可執行的完整範例」
>
> - 「用 FastAPI + python-jose 驗證 JWT，從 payload 取出 departments 與 clearance_level，傳給 Qdrant 的 metadata filter，完整實作企業 RAG 的 Identity-Aware Retrieval API」
>
> - 「用 OpenFGA 實作 RAG 文件的細粒度授權，定義 document / department / user 的關係模型，並在 Qdrant 取回 chunk 後做二次 can_read check，給我 Docker 啟動到 Python 呼叫的完整流程」

> 延伸閱讀：企業 AI 治理如何從 Policy 落地為 Technical Controls，請見 [企業 AI 治理框架：Agent 的 Least Privilege 與 RBAC/ABAC](./Blog/AI-Govs#tool-permission)。

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
    "@graph": [
      {
        "@type": "TechArticle",
        "mainEntityOfPage": {
          "@type": "WebPage",
          "@id": "https://deep-learning-101.github.io/RAG"
        },
        "headline": "RAG 實戰指南 2026：Chunking、多模態 Embedding、混合檢索與 Rerank 完整實作教學",
        "description": "2026 最新 RAG 技術實戰指南。從零打造高精準度本地端 RAG 系統，涵蓋 Chunking 策略、Qwen3-Embedding vs BGE-M3 選型、Gemini Embedding 2 與 Jina V5 Omni 多模態嵌入新選擇、Hybrid Search 混合檢索與 Qwen3-Reranker 排名優化、無向量 Visual RAG 架構，以及企業 RAG 的 Identity-Aware Retrieval 與 RBAC/ABAC 權限控管完整設計。",
        "image": "https://raw.githubusercontent.com/Deep-Learning-101/deep-learning-101.github.io/refs/heads/main/images/DeepLearning101-LOGO.png",
        "author": {
          "@type": "Person",
          "name": "TonTon Huang Ph.D.",
          "url": "https://twman.org/"
        },
        "publisher": {
          "@type": "Organization",
          "name": "Deep Learning 101, Taiwan",
          "url": "https://deep-learning-101.github.io/"
        },
        "datePublished": "2024-07-07T08:00:00+08:00",
        "dateModified": "2026-08-26T08:00:00+08:00",
        "keywords": "RAG, Retrieval-Augmented Generation, Chunking, Hybrid Search, Rerank, RAGAS, Embedding, Qwen3-Embedding, BGE-M3, Faithfulness, Context Recall, HyDE, Step-back Prompting, Visual RAG, 檢索增強生成, 幻覺, 冷啟動, RRF, 向量資料庫, Identity-Aware Retrieval, RBAC, ABAC, 企業 RAG 權限控管, metadata filter, 向量資料庫 ACL, OpenFGA, OPA, Qdrant, Weaviate, 多租戶隔離, Session Memory, 離職 RAG 安全, 換部門 知識庫 權限",
        "speakable": {
          "@type": "SpeakableSpecification",
          "cssSelector": ["h2", "h3", ".article-summary"]
        }
      },
      {
        "@type": "FAQPage",
        "mainEntity": [
          {
            "@type": "Question",
            "name": "企業導入 RAG 知識庫，如何避免機密資料外洩與 AI 幻覺？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "必須採用 100% 地端私有部署搭配開源模型（如 Qwen3-Embedding + Ollama/xinference），並在檢索層導入重排序（Rerank）、Chunking 策略優化與 LLM-Guard 零信任護欄。結合 Visual RAG 的無向量架構，可將幻覺率壓低至商用標準的 5% 以下，同時確保資料不出境。"
            }
          },
          {
            "@type": "Question",
            "name": "RAG 的 Chunking 策略有哪些？如何避免語意被截斷？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "主要有六大策略：①重疊切割（基礎兜底，chunk_size 的 10-20% 重疊）；②語意邊界切割（按句子/段落邊界，需 NLP 工具）；③句子視窗檢索（細粒度存儲，檢索後動態擴展上下文）；④父子切割（小塊精準檢索 + 大塊生成）；⑤命題化切割（LLM 分解為獨立命題，適合醫療/金融高精度場景）；⑥Contextual Retrieval（Anthropic 2024，為每個 chunk
  補全背景說明後再索引，可降低 49% 檢索失敗率）。"
            }
          },
          {
            "@type": "Question",
            "name": "Qwen3-Embedding 和 BGE-M3 哪個更適合繁體中文 RAG？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "Qwen3-Embedding-8B 目前是 MTEB 全球第 2、開源第 1（得分 68.12），C-MTEB 中文評測得分 72.88，全面超越 BGE-M3（64.63 / 68.31）。純文字檢索若預算允許，閉源的 google/gemini-embedding-001（MTEB 68.61）是性能天花板；若需同時處理圖文影音 PDF 的多模態場景，可評估 google/gemini-embedding-2（公開預覽，原生四模態同一向量空間）或 jinaai/jina-embeddings-v5-omni-small（開源，2026-05 發布，與舊索引 bit-identical 免重建）。BGE-M3 仍適合需要多向量（密集+稀疏）混合檢索的特殊場景。"
            }
          },
          {
            "@type": "Question",
            "name": "什麼是混合檢索（Hybrid Search）？為何比單純向量檢索更好？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "混合檢索結合向量語義搜尋（捕捉語義相似性）與 BM25 全文關鍵字搜尋（確保精確詞彙匹配）。向量搜尋可能忽略關鍵字，BM25 無法理解語義，兩者互補能大幅提升覆蓋率（Recall）。Anthropic 的 Contextual Retrieval 實驗顯示，混合檢索比純向量搜尋將 Top-20 失敗率降低約 49%。"
            }
          },
          {
            "@type": "Question",
            "name": "Rerank 重排序是什麼？Qwen3-Reranker 比 BGE-reranker 好在哪裡？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "Rerank 使用 Cross-Encoder 架構，將查詢與候選文件成對輸入模型進行深度相關性評分，是「從找得全到選得準」的關鍵一步。Qwen3-Reranker-4B/8B 在MTEB-R、CMTEB-R、MMTEB-R 及程式碼檢索上全面超越前代 BGE-reranker-v2-m3（例如 CMTEB-R 75.94 vs 72.16），已成為繁體中文 RAG 系統的首選 Reranker。"
            }
          },
          {
            "@type": "Question",
            "name": "無向量視覺 RAG (Visual RAG) 如何解決 PDF 表格與複雜排版問題？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "傳統 RAG 提取 PDF 表格時會將 2D 結構壓扁為 1D 純文字，導致欄位錯位與 LLM 幻覺。Vectorless Visual RAG 完全捨棄文字切塊與向量化，改以底層物理版面解析工具（如 OpenDataLoader + PyMuPDF）建立 JSON 目錄樹，記錄表格與圖片的精準 Bounding Box 與絕對頁碼。查詢時 LLM 先閱讀輕量目錄定位頁碼，再直接調用該頁高畫質原始截圖（JPEG/PNG）送給多模態 LLM（如 Gemini 2.5 Pro、GPT-4o）看圖作答，實現 0% 排版遺失。特別適合金融報告、法律合約、醫療 SOP 等需嚴格追蹤引用來源的場景。"
            }
          },
          {
            "@type": "Question",
            "name": "RAG 系統沒有歷史問答對，第一天怎麼建立評估基準？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "從 Chunk 對 LLM 合成問答對建立初始評估集，無需人工標註。5000 份文件約生成 2100 條可用問答對，耗時約 3 小時 LLM 調用。第 3–4 週引入 3–5 位領域專家標註 100–150 條難 case。文件質量治理（OCR 修正、去重、版本管理）可讓 Context Recall 從 0.67 提升到 0.79。上線後每月從真實對話篩 50–100 條替換合成數據，逐步讓評估集對齊真實用戶分佈。"
            }
          },
          {
            "@type": "Question",
            "name": "RAG 幻覺有哪些根源？如何系統性防控？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "RAG 幻覺有兩個根源：一是檢索失敗，LLM 靠自身訓練知識填充（例如知識庫寫「7 天退款」但未召回，LLM 答「30 天」）；二是檢索成功但 LLM 超範圍輸出，在 context 基礎上加入了文件未有的推斷。四道防線按成本遞增：①Prompt 強制四條約束規則；②Rerank 分數低於閾值（0.3–0.6）直接拒答；③生成後 LLM 引用核查（適合醫療/法律場景，成本翻倍）；④結構化 JSON 輸出附 source_ids 強制溯源。普通企業知識庫部署防線一+二即可，醫療合規建議四線全上。"
            }
          },
          {
            "@type": "Question",
            "name": "如何評估 RAG 系統的檢索層品質？Hit@K 和 MRR 怎麼用？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "Hit@K 衡量正確 chunk 是否出現在前 K 條結果中（Hit@5 < 0.7 說明 Embedding 或 Chunking 有問題；> 0.8 說明問題在生成層）。MRR（平均倒數排名）衡量正確 chunk 的排名位置，排名 1 得 1.0 分、排名 5 得 0.2 分，MRR < 0.5 說明 Rerank 效果不足。兩者需搭配使用：Hit@5 高但 MRR 低，說明找到了但排名靠後，送給 LLM 的 context 品質會下降。這兩個指標應在 RAGAS 評估之前先跑，確認檢索層合格再進入生成層診斷。"
            }
          },
          {
            "@type": "Question",
            "name": "企業 AI 系統為何需要「一鍵退場（Kill Switch）」機制？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "AI 模型會隨時間與新資料發生「模型漂移（Model Drift）」，導致決策偏差。一鍵退場機制讓管理層在 AI 出現歧視偏好或超出警戒紅線時，能瞬間切換回全人工審核模式，是防止演算法失控、確保企業法規遵循的最後安全底線，也是 AI 永續治理的核心要件。"
            }
          },
          {
            "@type": "Question",
            "name": "企業 RAG 知識庫如何做到不同部門的資料權限隔離？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "核心是 Identity-Aware Retrieval：文件入庫時附加 ACL metadata（department、clearance_level、expiry），查詢時根據用戶身份從 Identity Provider（如 Okta / Azure AD）即時取得 user.groups 與 user.clearance_level，再注入向量資料庫的 metadata pre-filter。向量相似度再高，只要 metadata filter 不過，chunk 就不會被取出放進 LLM context。護欄（Guardrail）是在 output 端過濾，無法阻止敏感 chunk 進入 context，因此 Access Control 必須在 retrieval 層發生。依隔離程度分四層：Metadata Filter（一般部門分層）、Partition/Namespace（財務 vs 工程）、獨立 Collection（董事會機密）、獨立 Vector DB 實例（法規要求實體隔離）。"
            }
          },
          {
            "@type": "Question",
            "name": "員工離職或換部門後，企業 RAG 的向量資料庫需要重建嗎？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "不需要重建 index。正確的設計是：Access Control 從 Identity Provider（如 Azure AD / Okta）即時取得，與 vector index 解耦。員工換部門只需更新 AD group 成員，下次查詢時 metadata filter 自動反映新權限；員工離職則停用帳號，token 失效後查詢就無法發起。文件設定 expiry 欄位後，過期文件也會被 filter 自動排除。唯一需要注意的真實風險是 Session Memory（多輪對話歷史）：若 session 沒有綁定 user_id 且結束後未清除，前輪取出的敏感 chunk 可能殘留在 context，需做 session-level ACL 管理。"
            }
          },
          {
            "@type": "Question",
            "name": "企業 RAG 權限控管有哪些開源工具？Qdrant、Weaviate 怎麼選？",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "向量資料庫方面：Qdrant 以 payload filter + multi-tenancy collection 實作 ACL，部署簡單，適合多數企業場景；Weaviate 有原生 RBAC 與 per-class auth，Enterprise 版功能更完整，適合需要細粒度角色管理的場景；Milvus/Zilliz 支援 RBAC 與 Partition Key，適合大規模資料；商用方面 Azure AI Search 有 security trimming 直接整合 Azure AD 群組，Google Vertex AI Search 繼承 Google Workspace 原生 ACL。授權決策引擎方面：OpenFGA（Google Zanzibar 開源實作）支援關係式授權，適合複雜組織結構；OPA（Open Policy Agent）用 Rego 語言寫政策，可嵌入任何服務；Casbin 支援 RBAC/ABAC/ReBAC，Python/Go/Java 皆有。推薦開源組合：Keycloak（身份認證）+ OpenFGA（授權決策）+ Qdrant（向量搜索）+ LlamaIndex identity middleware（RAG 框架）。"
            }
          }
        ]
      }
    ]
  }
  </script>