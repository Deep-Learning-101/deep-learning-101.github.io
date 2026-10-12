---
layout: default
title: Deep Learning 101 | 台灣首個深度學習社群 | LLM, CV, NLP & Speech 技術資源站
description: 台灣最早的深度學習社群 (Since 2016)。提供最新 AI 技術資源，包含 LLM 大語言模型、電腦視覺 (CV)、自然語言處理 (NLP) 與語音處理 (Speech) 的論文筆記與實作教學。
permalink: /
lang: zh-Hant
schema_type: Organization+Research+WebSite+service
service_type: AI Consulting
---

{% include header.html %}

---

## 📅 線上預約

歡迎提交你想在 Deep Learning 101 社群交流分享的議題，或訂閱候選議題，排程確定後會自動收到通知。

<iframe src="https://1999.twman.org/embed" width="100%" height="640" frameborder="0" style="border-radius:12px;border:1px solid #e5e7eb;"></iframe>

[➕ 提交新議題](https://1999.twman.org/submit) &nbsp;&nbsp; [📋 查看完整平台](https://1999.twman.org)

<script>
// 接收 embed iframe 的互動事件，轉送到本頁 GA4
window.addEventListener('message', function(e) {
  if (e.origin !== 'https://1999.twman.org') return;
  var d = e.data;
  if (!d || d.type !== 'dl101_ga') return;
  if (typeof gtag !== 'function') return;
  gtag('event', d.event, {
    event_category: 'embed_widget',
    topic_title:    d.title  || undefined,
    page_location:  window.location.pathname,
  });
});
</script>

---

{% include ai-share.html %}

---

現在市場上最容易出現的錯誤，是把「**有沒有 Agent**」當成企業 AI 數位轉型的 KPI。但對企業來說，真正應該問的並不是：

> 「我們公司要不要導入 AI Agent？」

而是：

> 「這個工作流程是否值得交給 AI？如果值得，企業**願意給**它多少資料、多少權限、多少計算資源，以及多少錯誤成本？」

這兩個問題完全不同。

---

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
  "datePublished": "2026-06-09",
  "dateModified": "2026-08-17",
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