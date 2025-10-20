---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

{% include header.html %}

---

{% include ai-share.html %}

---

# [Diffusion Model 完全解析：從原理、應用到實作 (AI 圖像生成)；ComfyUI + Stable Diffusion + FLUX](https://deep-learning-101.github.io/)

**作者**：[TonTon Huang Ph.D.](https://twman.org/)
**日期**：2024年11月1日  
**原文網址**：[https://blog.twman.org/2024/11/diffusion.html](https://blog.twman.org/2024/11/diffusion.html)

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

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/diffusion"
  },
  "headline": "Diffusion Model 完全解析：從原理、應用到實作 (AI 圖像生成)",
  "description": "深入探討擴散模型（Diffusion Model）的原理、應用與實作，並介紹 ComfyUI、Stable Diffusion、FLUX、Wan2.1 等相關工具與模型，提供 AI 圖像與影片生成的實務指南。",
  "image": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg",
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
      "url": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg"
    }
  },
  "datePublished": "2024-11-01",
  "dateModified": "2024-11-01",
  "keywords": "Diffusion Model, AI圖像生成, ComfyUI, Stable Diffusion, FLUX, Wan2.1, HunyuanVideo, Text-to-Image, 擴散模型"
}
</script>