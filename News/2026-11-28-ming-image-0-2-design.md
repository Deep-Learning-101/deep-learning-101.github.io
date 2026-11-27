---
layout: default
title: "螞蟻開源 Ming-Image-0.2-Design：6B 設計專用生圖模型，拿下開源權重榜第一"
description: "螞蟻集團 InclusionAI 開源 6B 設計專用文生圖模型 Ming-Image-0.2-Design，MIT 授權，在 AA-Image-T2I 榜以 Elo 1,095 拿下開源權重第一。強項是複雜構圖、文字渲染與可編輯圖層拆解，本篇整理已查證的規格、榜單與限制。"
permalink: /News/2026-11-28-ming-image-0-2-design
date: 2026-11-28
lang: zh-Hant
keywords: ["Ming-Image", "InclusionAI", "螞蟻集團", "文生圖", "開源模型", "設計"]
tags: ["AI快訊", "CV"]
---

{% include header.html %}

---

{% include ai-share.html %}

---

**作者**：[TonTon Huang Ph.D.](https://twman.org)
**首次發布**：2026-11-28 (台北時間)
**事件時間**：2026-11-26，權重開源上架

# 螞蟻開源 Ming-Image-0.2-Design：6B 設計專用生圖模型，拿下開源權重榜第一

開源生圖模型一直有個老問題：圖生得出來，但構圖亂、文字寫錯，設計師拿到還得重畫。這次螞蟻開源的模型專攻設計場景，而且把圖層拆解做到可以逐字編輯，方向跟通用生圖不一樣。

## 發生什麼事

螞蟻集團 InclusionAI 在 11 月 26 日開源 Ming-Image-0.2-Design，6B 參數的擴散 Transformer (DiT)，MIT 授權，權重上架 Hugging Face 與 ModelScope。它是設計專用模型：電商主圖、介面草稿、空間渲染這類工作，強調複雜構圖、光線與文字渲染；圖層拆解功能可以把一張設計拆成獨立可編輯的透明圖層，細到單一字元，方便後續編輯與動畫。搭配自家的 Ling-3.2-flash 能把設計概念轉成可互動的網頁，另有工作流可以把生成的投影片圖片還原成可編輯的 PowerPoint。

榜單方面：在 Artificial Analysis 的 AA-Image-T2I v2.1 榜上以 Elo 1,095 (95% 區間 1,083 至 1,107、4,936 個樣本) 拿下開源權重第一、總榜第 24 名 (榜單快照為 2026 年 11 月 25 日)。預設輸出 2048×2048、12 步採樣，支援 vLLM-Omni 部署。原始資料見 [HF 模型卡](https://huggingface.co/inclusionAI/Ming-Image-0.2-Design)。

## 為什麼重要

第一，開源權重第一名換人做，而且是專攻設計的模型，不是通用模型順便會畫圖：對做電商視覺、活動主圖與 UI 草稿的團隊，這是第一個授權乾淨 (MIT)、規格又對症的開源選項。第二，圖層拆到字元級是真正的分水嶺：生圖從「交一張扁平成品」變成「交一份可編輯的半成品」，設計師的後製成本完全不同，這比榜單名次更值得注意。第三，限制照實講：官方驗證配置是一張 80GB 顯存的 GPU，不是家用電腦能跑的體量；負責圖層拆解的 Layer 版要下週才開源，現在這版先收的是 Design 本體。

## 接下來看什麼

Layer 版開源後，圖層拆解的實際品質與速度；社群用 vLLM-Omni 部署的實測回報；以及 Artificial Analysis 榜單後續快照裡，它跟閉源模型的名次變化。

## 延伸閱讀

- [電腦視覺 (Computer Vision) 深度頁](https://deep-learning-101.github.io/Computer-Vision)：本站的文生圖與視覺模型整理，本案的完整條目 (架構、榜單全表與痛點分析) 之後也會收進這裡。

## 更新紀錄

- 2026-11-28：首次發布。
