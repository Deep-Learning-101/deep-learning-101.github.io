---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

# [Diffusion Model 完全解析：從原理、應用到實作 (AI 圖像生成)；ComfyUI + Multimodal + Stable Diffusion + FLUX](https://deep-learning-101.github.io/)

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)   
**日期**：2024年11月1日  
**原文網址**：[https://blog.twman.org/2024/11/diffusion.html](https://blog.twman.org/2024/11/diffusion.html)

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

本文深入探討了擴散模型（Diffusion Model）的原理、應用場景與實作經驗，並介紹了多種相關工具與模型，包括 Stable Diffusion、ComfyUI、FLUX、Wan2.1、HunyuanVideo 以及 Step-Video-TI2V 等，提供了豐富的實務操作建議與資源。

---

## 主要內容摘要

### 1. 擴散模型（Diffusion Model）概述

- **Stable Diffusion**：基於擴散模型的圖像生成技術，支援文本到圖像（Text-to-Image）、圖像修改（Image-to-Image）、修補（Inpainting）與擴展圖像（Outpainting）等功能。
- **應用場景**：廣泛應用於藝術創作、影像編輯、影片生成等多種領域。

### 2. ComfyUI 與多模態應用

- **ComfyUI**：一個模組化的圖形化界面，方便用戶構建和管理擴散模型的工作流程。
- **多模態應用**：結合文字、圖像、影片等多種資料形式，實現更豐富的生成效果。

### 3. FLUX 模型介紹

- **FLUX.1**：由 Black Forest Labs 推出的文本生成圖像模型，包含三個版本：
  - **FLUX.1 [pro]**：商業用途設計的閉源模型。
  - **FLUX.1 [dev]**：開源的引導蒸餾模型，適用於非商業應用。
  - **FLUX.1 [schnell]**：專為本地開發和個人使用設計。
- **FLUX.1 工具**：增強模型可控制性與可操作性的工具集，包括：
  - **FLUX.1 Fill**：根據文字描述和遮罩編輯圖像。
  - **FLUX.1 Depth**：從輸入影像和文字提示中提取深度圖。
  - **FLUX.1 Canny**：從輸入影像和文字提示中提取 Canny 邊緣。
  - **FLUX.1 Redux**：混合和重新建立輸入影像和文字提示的適配器。

### 4. Wan2.1 模型介紹

- **Wan2.1**：阿里雲通義萬象開源的多功能視訊生成模型，支援：
  - 文字轉影片（Text-to-Video）
  - 圖片轉影片（Image-to-Video）
  - 影片編輯
  - 影片轉音訊
- **技術特點**：
  - 採用 VAE + 擴散變換器（DiT），增強時間建模與場景理解。
  - 支援 1080p 高清解析度，並可同時生成動態字幕及多語言配音。
  - 2025年1月登頂 VBench 榜首，性能領先全球。

### 5. HunyuanVideo 模型介紹

- **HunyuanVideo-I2V**：騰訊混元發布的圖生視頻模型，基於 HunyuanVideo 文生視訊基礎模型，擴展至影像到視訊的生成任務。
- **部署方式**：
  - 提供 LoRA 訓練程式碼，用於客製化特效生成。
  - 可在 ComfyUI 中進行部署與使用。

### 6. Step-Video-TI2V 模型介紹

- **Step-Video-TI2V**：階躍星辰開源的圖生視訊模型，支援圖片轉影片的生成任務。
- **部署挑戰**：
  - 高顯存需求，建議使用 32GB 以上的 GPU。
  - 需調整系統的 swap 設定，以避免記憶體不足的問題。

---

## 結語

擴散模型在 AI 圖像與影片生成領域展現出強大的潛力，透過結合多種工具與模型，如 Stable Diffusion、ComfyUI、FLUX、Wan2.1、HunyuanVideo 及 Step-Video-TI2V 等，能夠實現更豐富、多樣化的創作與應用。實作過程中需注意硬體資源的配置與系統設定，以確保模型的順利運行。

---

> 📖 如需進一步了解，請參閱原文：  
> [https://blog.twman.org/2024/11/diffusion.html](https://blog.twman.org/2024/11/diffusion.html)
