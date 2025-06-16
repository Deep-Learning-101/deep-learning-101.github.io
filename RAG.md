---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

{% include header.html %}

---

{% include ai-share.html %}

---

# [檢索增強生成 (Retrieval-Augmented Generation, RAG) 不是萬靈丹：檢索增強生成的挑戰與優化技巧](https://deep-learning-101.github.io/)

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**日期**：2024年7月7日  
**原文網址**：[https://blog.twman.org/2024/07/RAG.html](https://blog.twman.org/2024/07/RAG.html)

---

## 文章概述

作者分享了在實作 RAG（Retrieval-Augmented Generation）過程中遇到的挑戰與優化技巧，並強調 RAG 並非萬靈丹，需根據實際需求進行適當的設計與調整。

---

## 主要內容摘要

### 1. 為何選擇 RAG？

- **聚焦私有資料**：RAG 能專注於私有資料的檢索與生成，提升回應的相關性。
- **結合檢索與生成**：透過檢索相關資料並結合生成模型，提供更精確的答案。

### 2. 本地端部署 RAG 的工具

- **Ollama**：簡化大模型的部署與執行，支援嵌入生成。
- **Xinference**：提供更方便的操作介面，適合本地端使用。

### 3. 文檔處理與嵌入生成

- **文檔切分**：避免直接將整份 PDF 或 PPT 輸入，應先進行適當的切分與整理。
- **工具推薦**：
  - [MinerU](https://github.com/opendatalab/MinerU)：一站式資料擷取工具，支援 PDF、網頁、電子書等格式。
  - [Omniparse](https://github.com/adithya-s-k/omniparse)：支援多種資料格式的解析與優化，適用於 GenAI 框架。
  - [PyMuPDF](https://zhuanlan.zhihu.com/p/pyMuPDF)：Python 處理 PDF 的工具。
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 與 [RapidOCR](https://rapidai.github.io/RapidOCR/)：適用於掃描檔的 OCR 處理。

### 4. 檢索策略與排序機制

- **兩階段檢索系統**：
  - **嵌入模型（Embedding Model）**：快速篩選相關文件。
  - **重新排序器（Reranker）**：精確排序，提升結果的相關性。
- **檢索方式**：
  - **向量檢索**：透過生成查詢嵌入並查詢與其向量表示最相似的文字分段。
  - **全文檢索**：索引文件中的所有詞彙，允許使用者查詢任意詞彙。
  - **混合檢索**：結合向量與全文檢索，並使用 Reranker 進行重新排序。

### 5. 知識圖譜與 Ontology 的應用

- **GraphRAG**：結合知識圖譜與 RAG，提升生成內容的結構性與準確性。
- **Ontology 的重要性**：強調在知識管理與檢索中的角色，並分享作者在 OWL.Manchester 的經驗。

---

## 結語

RAG 提供了一種結合檢索與生成的強大方法，但並非適用於所有情境。實作時需根據實際需求選擇合適的工具與策略，並注意資料處理與模型部署的細節，才能發揮其最大效益。

---

> 📖 如需進一步了解，請參閱原文：  
> [https://blog.twman.org/2024/07/RAG.html](https://blog.twman.org/2024/07/RAG.html)
