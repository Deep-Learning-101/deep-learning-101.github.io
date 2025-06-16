---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

{% include header.html %}

---

{% include ai-share.html %}

---

# [解析探索大型語言模型：模型發展歷史、訓練及微調技術的 VRAM 估算](https://deep-learning-101.github.io/)

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)   
**日期**：2023年4月12日  
**原文網址**：[https://blog.twman.org/2023/04/GPT.html](https://blog.twman.org/2023/04/GPT.html)

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
