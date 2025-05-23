---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

# [解析探索大型語言模型：模型發展歷史、訓練及微調技術的 VRAM 估算](https://deep-learning-101.github.io/)

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)   
**日期**：2023年4月12日  
**原文網址**：[https://blog.twman.org/2023/04/GPT.html](https://blog.twman.org/2023/04/GPT.html)

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

本文深入探討大型語言模型（LLM）的發展歷史、訓練與微調技術，並詳細解析在不同精度與訓練策略下，所需的 GPU VRAM 估算方法，為從業者提供實作參考。

---

## 主要內容摘要

### 1. 模型參數與 VRAM 估算基礎

- **參數數量與記憶體需求**：
  - 1B（10億）參數約需 4GB VRAM（FP32 精度）。
  - 精度降低（如 FP16）則記憶體需求減半。
- **常見精度格式**：
  - FP32（單精度）：每參數佔 4 bytes。
  - FP16/BF16（半精度）：每參數佔 2 bytes。
  - INT8（8位整數）：每參數佔 1 byte，常用於推理階段。

### 2. 訓練與微調的 VRAM 需求估算

#### 全參數訓練（Full Parameter Training）

- **FP32 精度**：
  - 模型權重：4X GB
  - 梯度：4X GB
  - 優化器狀態（如 AdamW）：8X GB
  - **總計**：16X GB + 啟動值與其他開銷
- **FP16/BF16 精度**：
  - 模型權重：2X GB
  - 梯度：2X GB
  - 優化器狀態：8X GB
  - **總計**：12X GB + 啟動值與其他開銷

#### 全參數微調（Full Fine-tuning）

- 與全參數訓練相似，但通常 batch size 較小，啟動值需求較低。
- **估算**：
  - 7B 模型：約 100–140 GB VRAM
  - 70B 模型：超過 1 TB VRAM

#### LoRA 微調（Low-Rank Adaptation）

- 僅訓練少量適配器參數，凍結原始模型大部分參數。
- **估算**：
  - 7B 模型：約 16–24 GB VRAM
  - 70B 模型：約 140–200 GB VRAM

---

## 實作經驗分享

作者分享了將 Deep Learning Book 的 PDF 進行重點摘要，並對影片進行語音辨識與逐字稿生成的經驗，展示了大型語言模型在實際應用中的潛力與挑戰。

---

## 結語

大型語言模型的訓練與微調對硬體資源有著極高的需求，透過合理的精度選擇與訓練策略，可以有效降低 VRAM 的使用，提升訓練效率。本文提供的估算方法與實作經驗，對於從事 LLM 開發與應用的從業者具有重要參考價值。

---

> 📖 如需進一步了解，請參閱原文：  
> [https://blog.twman.org/2023/04/GPT.html](https://blog.twman.org/2023/04/GPT.html)
