---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

# [檢索增強生成 (Retrieval-Augmented Generation, RAG) 不是萬靈丹：檢索增強生成的挑戰與優化技巧](https://deep-learning-101.github.io/)

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**日期**：2024年7月7日  
**原文網址**：[https://blog.twman.org/2024/07/RAG.html](https://blog.twman.org/2024/07/RAG.html)

---

<p align="center">
  <strong>Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101</strong>  
</p>
<p align="center">
  AI是一條孤獨且充滿惶恐及未知的旅程，花俏絢麗的收費課程或活動絕非通往成功的捷徑。<br>
  衷心感謝當時來自不同單位的AI同好參與者實名分享的寶貴經驗；如欲移除資訊還請告知。<br>
  由 <a href="https://www.twman.org/" target="_blank">TonTon Huang Ph.D.</a> 發起，及其當時任職公司(台灣雪豹科技)無償贊助場地及茶水點心。<br>
</p>  
<p align="center">
  <a href="https://huggingface.co/spaces/DeepLearning101/Deep-Learning-101-FAQ" target="_blank">
    <img src="https://github.com/Deep-Learning-101/.github/blob/main/images/DeepLearning101.JPG?raw=true" alt="Deep Learning 101" width="180"></a>
    <a href="https://www.buymeacoffee.com/DeepLearning101" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-red.png" alt="Buy Me A Coffee" style="height: 100px !important;width: 180px !important;" ></a>
</p>
<p align="center">
  <a href="https://www.youtube.com/@DeepLearning101" target="_blank">YouTube</a> |
  <a href="https://www.facebook.com/groups/525579498272187/" target="_blank">Facebook</a> |
  <a href="https://deep-learning-101.github.io/"> 回 GitHub Pages</a> |
  <a href="http://DeepLearning101.TWMAN.ORG" target="_blank">網站</a> |
  <a href="https://huggingface.co/DeepLearning101" target="_blank">Hugging Face Space</a>
</p>

---

<div align="center">

<table>
  <tr>
    <td align="center"><a href="https://deep-learning-101.github.io/Large-Language-Model">大語言模型</a></td>
    <td align="center"><a href="https://deep-learning-101.github.io/Speech-Processing">語音處理</a></td>
    <td align="center"><a href="https://deep-learning-101.github.io/Natural-Language-Processing">自然語言處理</a></td>
    <td align="center"><a href="https://deep-learning-101.github.io//Computer-Vision">電腦視覺</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper?tab=readme-ov-file#llm">Large Language Model</a></td>
    <td><a href="https://github.com/Deep-Learning-101/Speech-Processing-Paper">Speech Processing</a></td>
    <td><a href="https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper">Natural Language Processing, NLP</a></td>
    <td><a href="https://github.com/Deep-Learning-101/Computer-Vision-Paper">Computer Vision</a></td>
  </tr>
</table>

</div>

---

<details>
<summary>手把手帶你一起踩 AI 坑</summary>

<h3><a href="https://blog.twman.org/p/deeplearning101.html" target="_blank">手把手帶你一起踩 AI 坑</a>：<a href="https://www.twman.org/AI" target="_blank">https://www.twman.org/AI</a></h3>

<ul>
  <li>
    <b><a href="https://blog.twman.org/2025/03/AIAgent.html" target="_blank">避開 AI Agent 開發陷阱：常見問題、挑戰與解決方案</a></b>：<a href="https://deep-learning-101.github.io/agent" target="_blank">探討多種 AI 代理人工具的應用經驗與挑戰，分享實用經驗與工具推薦。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/08/LLM.html" target="_blank">白話文手把手帶你科普 GenAI</a></b>：<a href="https://deep-learning-101.github.io/GenAI" target="_blank">淺顯介紹生成式人工智慧核心概念，強調硬體資源和數據的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/09/LLM.html" target="_blank">大型語言模型直接就打完收工？</a></b>：<a href="https://deep-learning-101.github.io/1010LLM" target="_blank">回顧 LLM 領域探索歷程，討論硬體升級對 AI 開發的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/07/RAG.html" target="_blank">檢索增強生成(RAG)不是萬靈丹之優化挑戰技巧</a></b>：<a href="https://deep-learning-101.github.io/RAG" target="_blank">探討 RAG 技術應用與挑戰，提供實用經驗分享和工具建議。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/02/LLM.html" target="_blank">大型語言模型 (LLM) 入門完整指南：原理、應用與未來</a></b>：<a href="https://deep-learning-101.github.io/0204LLM" target="_blank">探討多種 LLM 工具的應用與挑戰，強調硬體資源的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2023/04/GPT.html" target="_blank">什麼是大語言模型，它是什麼？想要嗎？(Large Language Model，LLM)</a></b>：<a href="https://deep-learning-101.github.io/GPU" target="_blank">探討 LLM 的發展與應用，強調硬體資源在開發中的關鍵作用。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/11/diffusion.html" target="_blank">Diffusion Model 完全解析：從原理、應用到實作 (AI 圖像生成)</a></b>；<a href="https://deep-learning-101.github.io/diffusion" target="_blank">深入探討影像生成與分割技術的應用，強調硬體資源的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/02/asr-tts.html" target="_blank">ASR/TTS 開發避坑指南：語音辨識與合成的常見挑戰與對策</a></b>：<a href="https://deep-learning-101.github.io/asr-tts" target="_blank">探討 ASR 和 TTS 技術應用中的問題，強調數據質量的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2021/04/NLP.html" target="_blank">那些 NLP 踩的坑</a></b>：<a href="https://deep-learning-101.github.io/nlp" target="_blank">分享 NLP 領域的實踐經驗，強調數據質量對模型效果的影響。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2021/04/ASR.html" target="_blank">那些語音處理踩的坑</a></b>：<a href="https://deep-learning-101.github.io/speech" target="_blank">分享語音處理領域的實務經驗，強調資料品質對模型效果的影響。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2020/05/DeepLearning.html" target="_blank">手把手學深度學習安裝環境</a></b>：<a href="https://deep-learning-101.github.io/101" target="_blank">詳細介紹在 Ubuntu 上安裝深度學習環境的步驟，分享實際操作經驗。</a>
  </li>
</ul>

</details>

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
