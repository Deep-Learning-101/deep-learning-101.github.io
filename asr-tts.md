---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

{% include header.html %}

---

{% include ai-share.html %}

---

# [ASR/TTS 開發避坑指南：語音辨識與合成的常見挑戰與對策；那些ASR和TTS可能會踩的坑](https://deep-learning-101.github.io/)

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)   
**日期**：2024年2月25日  
**原文網址**：[https://blog.twman.org/2024/02/asr-tts.html](https://blog.twman.org/2024/02/asr-tts.html)

---

## 文章概述

本文分享了在開發自動語音辨識（ASR）與文字轉語音（TTS）應用時，常見的挑戰與解決策略，特別是針對中文語音處理的實務經驗與工具應用。

---

## 主要內容摘要

### 1. Whisper 模型的應用與挑戰

- **Whisper**：OpenAI 推出的多語言語音辨識模型，支援中文語音轉文字。
- **挑戰**：
  - 中文標點符號處理效果不佳，需透過其他模型輔助。
  - 需要進行微調以提升特定場景的辨識準確度。

### 2. 開源工具與模型的應用

- **faster-whisper**：Whisper 的高效實現版本，提升推理速度。
- **WhisperX**：在 faster-whisper 基礎上，加入時間戳記功能。
- **BELLE-2**：提供微調過的 Whisper 模型，提升中文辨識效果。
- **Whisper-Finetune**：提供 Whisper 模型的微調方法與工具。
- **WhisperStreaming**：支援長時間語音的串流轉錄與翻譯。
- **WhisperSpeech / WhisperLive / WhisperFusion**：結合 ASR 與 LLM，實現語音問答與合成。

### 3. 語音資料的處理與準備

- **語音切割工具**：
  - **Audio-Slicer**：精準剪裁音頻，製作訓練語料。
  - **Ultimate Vocal Remover 5 (UVR5)**：分離人聲與背景音。
  - **Denoiser**：Facebook Research 提供的語音去噪工具。
- **語音資料格式**：需製作符合 Whisper 微調需求的語料格式。

### 4. 中文語音辨識的替代方案

- **FunASR**：阿里巴巴達摩院推出的中文語音辨識模型，訓練於 6 萬小時的中文語料，適合中文應用場景。

### 5. ASR 辨識後的錯誤修正

- **FastCorrect / FastCorrect2**：微軟提供的語音辨識錯誤自動修正工具。
- **AdapterASR**：微軟提供的 ASR 模型適配器，提升特定場景的辨識效果。
- **pycorrector**：中文文本糾錯工具，適用於 ASR 結果的後處理。

---

## 結語

在開發 ASR 與 TTS 應用時，需考量模型選擇、語音資料處理、微調策略與後處理方法，以提升系統的整體效能與準確度。透過結合多種工具與模型，能有效應對中文語音處理的挑戰，實現更自然流暢的語音應用。

---

> 📖 如需進一步了解，請參閱原文：  
> [https://blog.twman.org/2024/02/asr-tts.html](https://blog.twman.org/2024/02/asr-tts.html)
