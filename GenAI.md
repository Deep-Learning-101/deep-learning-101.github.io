---
layout: default
title: 白話文手把手帶你科普 GenAI | LLM, RAG, 微調與多模態核心概念
description: 淺顯易懂地介紹生成式人工智慧（GenAI）的核心技術名詞，包含大語言模型 (LLM)、提示詞 (Prompt)、檢索增強生成 (RAG)、微調 (Fine-Tuning)、代理人 (Agentic) 與多模態，帶你快速掌握 AI 發展脈絡。
permalink: /GenAI
lang: zh-Hant
schema_type: article
tags: ["GenAI", "LLM", "RAG", "入門教學"]
---

{% include header.html %}

---

{% include ai-share.html %}

---

# 🚀 白話文手把手帶你科普 GenAI

生成式人工智慧（Generative AI）正在改變世界。本頁面以最淺顯易懂的「白話文」，為非工程背景的讀者梳理近一年來最核心的 AI 技術名詞與運作邏輯。

> **作者**：[TonTon Huang Ph.D.](https://twman.org/)  
> **原文出處**：[白話文手把手帶你科普 GenAI 的 LLM、Prompt... (發布於 2024/08/11)](https://blog.twman.org/2024/08/LLM.html)

---

## 💸 訓練 GenAI 的門檻：算力與資本

在進入技術名詞前，必須先理解 AI 發展背後的巨大硬體資源戰。以 Meta 開源的 **Llama 3.1 405B** 模型為例：
* **使用硬體**：16,384 張 Nvidia H100 80GB GPU
* **訓練耗時**：54 天
* **預估硬體總成本**：高達近 **1 億美金**

這顯示出龐大的「算力 (Compute)」與「數據 (Data)」是當今推進 GenAI 極限的最重要護城河。

---

## 🧠 核心技術與名詞解析

### 1. 大型語言模型（LLM）
LLM (Large Language Model) 是當代 GenAI 的大腦基礎。它透過閱讀海量文本進行訓練，從而擁有理解與生成自然語言的強大能力。
* 📌 **延伸資源**：[LLM 中文資源整理 - Awesome List](https://blog.twman.org/2024/08/LLM.html)

### 2. 提示詞工程（Prompt Engineering）
Prompt 是我們輸入給 LLM 的文字指令或上下文。良好的 Prompt 設計，能引導模型準確收斂，大幅提升輸出品質，是操作 AI 最基礎也最重要的技巧。

### 3. 檢索增強生成（RAG）
RAG (Retrieval-Augmented Generation) 是一種將「資訊檢索」與「文本生成」結合的技術。它允許模型在回答前，先去外部資料庫尋找最新或私有知識，再進行回答。
* **適用場景**：企業內部知識庫客服、需要即時最新資訊的應用。
* **優勢對比**：相比於重新訓練模型，RAG 具備實時性強、能有效減少模型「幻覺 (Hallucination)」的優點。

### 4. 模型微調（Fine-Tuning）
在基礎的預訓練模型上，使用特定的資料集進行二次訓練。這能讓模型專精於某個領域（例如醫療、法律）或特定任務。
* **成本優勢**：微調開源模型的成本，往往只有每次都呼叫頂級商用 API 的一小部分。

### 5. 功能調用（Function Calling）
模型不再只是單純的聊天機器人，它能根據使用者的語義，判斷是否需要「呼叫外部工具」。
* **運作邏輯**：接收指令 → 模型決定執行某個函數（如查天氣、撈資料庫） → 回傳結果並生成最終答案。

### 6. 工作流程（Workflow）
將多個 Function Calling 串聯起來，讓模型能自動規劃步驟與邏輯，完成一系列複雜的連續任務。

### 7. 代理人架構（Agentic）
這是目前 AI 發展的熱門趨勢。Agent 是一個結合了多種工具的系統，它具備**自主決策、任務拆解與規劃**的能力，能像一個真實的數位助理一樣，為你自動完成多步驟的任務。

### 8. 多模態（Multimodal）
打破單一文字的限制，讓 AI 系統能同時處理、理解並生成「文本、圖像、音訊、影片」等多種資料類型，具備跨感官的綜合感知能力。

---

> 💡 **結語**：想學好 GenAI，別只停留在「怎麼用 ChatGPT」。從底層的 Prompt 到 RAG，再到未來的 Agent 架構，一步步理解背後的技術脈絡，才是真正的升級之道！

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/GenAI"
  },
  "headline": "白話文手把手帶你科普 GenAI | LLM, RAG, 微調與多模態核心概念",
  "description": "淺顯易懂地介紹生成式人工智慧（GenAI）的核心技術名詞，包含大語言模型 (LLM)、提示詞 (Prompt)、檢索增強生成 (RAG)、微調 (Fine-Tuning)、代理人 (Agentic) 與多模態。",
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
  "datePublished": "2024-08-11",
  "dateModified": "2026-03-29",
  "keywords": "Generative AI, GenAI, LLM, RAG, Fine-Tuning, Prompt, Function Calling, Agentic, 多模態, Multimodal"
}
</script>