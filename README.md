---
layout: default
title: Deep Learning 101 | 台灣首個深度學習社群 | LLM, CV, NLP & Speech 技術資源站
description: 台灣最早的深度學習社群 (Since 2016)。提供最新 AI 技術資源，包含 LLM 大語言模型、電腦視覺 (CV)、自然語言處理 (NLP) 與語音處理 (Speech) 的論文筆記與實作教學。
permalink: /
lang: zh-Hant
schema_type: service
service_type: AI Consulting
---

<style>
/* 核心卡片網格系統 */
.tech-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 30px;
  margin-bottom: 40px;
}

/* 單張卡片樣式 */
.tech-card {
  background: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 24px;
  text-decoration: none !important; /* 蓋掉預設連結底線 */
  color: #333 !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  display: flex;
  flex-direction: column;
}

/* 滑鼠移過去的動態效果 */
.tech-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 20px rgba(0,0,0,0.1);
  border-color: #007bff;
}

.tech-card h3 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 1.25rem;
  color: #1a1a1a;
  border-bottom: none; /* 移除可能存在的底線 */
}

.tech-card p {
  margin: 0;
  font-size: 0.95rem;
  color: #666;
  line-height: 1.5;
  flex-grow: 1; /* 讓文字自動推開下方內容 */
}

.tech-card .tags {
  margin-top: 16px;
  font-size: 0.8rem;
  color: #007bff;
  font-weight: bold;
}
</style>

---

{% include header.html %}

---

<div style="text-align: center; margin: 40px 0;">
  <a href="/UPDATE" style="display: inline-block; background-color: #007bff; color: white; padding: 14px 28px; border-radius: 50px; font-weight: bold; text-decoration: none; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(0,123,255,0.3); transition: 0.3s;">
    🚀 點此查看 2026 最新技術迭代快訊 (Changelog)
  </a>
</div>

---

{% include ai-share.html %}

---

# Deep Learning 101 Meetup

Deep Learning 101 於 2016/11/11 在台北 101 大樓 83F 成立，是台灣最具開創性的深度學習同好會。這裡不僅匯集了我們歷年的 Meetup 紀錄，更是社群共同維護的 AI 演算法與開源資源匯整中心。

👉 [查看 Deep Learning 101 歷年所有實體 Meetup 影像與逐字稿](/meetups)

## 🧭 探索核心技術領域 (Knowledge Base)

<div class="tech-grid">

  <a href="/Large-Language-Model" class="tech-card">
    <h3>🤖 大語言模型 (LLM) & Agents</h3>
    <p>涵蓋 RAG 防幻覺實作、多智能體框架、端側小模型 (SLM) 選型與微調技術。</p>
    <div class="tags">#vLLM #AutoGen #RAG</div>
  </a>

  <a href="/Computer-Vision" class="tech-card">
    <h3>👁️ 電腦視覺 (CV)</h3>
    <p>異常檢測 (AOI)、高精度 OCR (MinerU)、物件偵測與擴散模型生成解析。</p>
    <div class="tags">#YOLO #Diffusion #OCR</div>
  </a>

  <a href="/Speech-Processing" class="tech-card">
    <h3>🎤 語音處理 (Speech AI)</h3>
    <p>免切片語音辨識、即時人聲分離、語音增強去噪與 TTS 合成實戰。</p>
    <div class="tags">#ASR #TTS #VibeVoice</div>
  </a>

  <a href="/cyber/LLM-Offense" class="tech-card">
    <h3>⚔️ AI 安全攻防與護欄</h3>
    <p>LLM 越獄防禦、自動化紅隊演練 (FuzzyAI) 與企業級安全護欄架構部署。</p>
    <div class="tags">#RedTeaming #LlamaGuard</div>
  </a>

</div>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "WebPage",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/"
  },
  "name": "Deep Learning 101 Meetup",
  "description": "Deep Learning 101 是臺灣領先的深度學習同好會，自 2016 年 11 月 11 日起在台北 101 舉辦活動。此頁面彙整了歷年來的技術分享主題、講者資訊、影片連結與活動摘要。",
  "publisher": {
    "@type": "Organization",
    "name": "Deep Learning 101, Taiwan",
    "logo": {
      "@type": "ImageObject",
      "url": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg"
    }
  },
  "datePublished": "2026-03-29",
  "dateModified": "2026-03-29",
  "about": {
    "@type": "EventSeries",
    "name": "Deep Learning 101 Meetup",
    "startDate": "2016-11-11",
    "location": {
      "@type": "Place",
      "name": "Taipei 101",
      "address": "Taipei, Taiwan"
    }
  }
}
</script>