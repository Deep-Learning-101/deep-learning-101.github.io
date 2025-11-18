---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---


{% include header.html %}

---

{% include ai-share.html %}

---

<div class="podcast-grid">
  <div class="podcast-cell">
    🎵 不聽可惜的 NotebookLM<br>
    <audio controls>
      <source src="../notebooklm-mp3/AIBeginner.mp3" type="audio/mpeg">
    </audio>
    <p>企業導入 AI 的聽君一席話，如聽一席話入門全攻略</p>
  </div>

  <div class="podcast-cell">
    🎵 不聽可惜的 NotebookLM<br>
    <audio controls>
      <source src="../notebooklm-mp3/AIBeginnerMistakes.mp3" type="audio/mpeg">
    </audio>
    <p>認真 企業導入 AI 全攻略</p>
  </div>

  <div class="podcast-cell">
    <div class="video-container">
      <iframe
        src="https://www.youtube.com/embed/IZGqfbj5AF4"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen>
      </iframe>
    </div>
  </div>

  <div class="podcast-cell">
    <div class="video-container">
      <iframe
        src="https://www.youtube.com/embed/59ojQx_dLFo"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen>
      </iframe>
    </div>
  </div>
</div>

---

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**日期**：2025年07月21日更新

---

# 企業 AI新賽局 入門策略實踐路線圖：從策略到治理，避開致命陷阱

第一波夯起來的時間估計是 2016年 Google DeepMind 的 Alpha Go 吧，但深度學習確實很難實際的被應用到各行各業，特別是台灣，在那幾年，做個規則式的聊天機器人，或者廣告點擊，可能就會先用新聞稿把自己吹成 AI 大神或新創了。第二波就是 Open AI 橫空出世的ChatGPT了，然後四處都能看到大家都說自己有GenAI，可惜可能連個幾張消費型的4090等級的GPU都沒有，更別說已屬上世代的V100 或者連 GCP、AWS跟Azure的GPU都沒啟用設定過吧？就目前體驗，雖然已有不少國外的新聞或資訊報導可能碰上的問題。  

但這裡是**『台灣』**，通常更可能發生：
- 高層心血來潮的玻璃心：因為新聞都在報 GenAI，所以不跟著 Gen 一下好像都對不起自己的職級；BUT，高層完全不懂也不想懂更不想花錢，就是做就對了 !
- 昂貴設備的高風險投資：據記憶，曾處理過 A100*4 以及 RTX 6000 Ada * 8 (皆約$330萬新台幣)，這是地端自建價格；至於[雲端API請參考這](https://deep-learning-101.github.io/Blog/TW-LLM-Benchmark#%E7%B8%BD%E9%AB%94%E6%88%B0%E7%95%A5%E6%AF%94%E8%BC%83%E4%B8%89%E5%A4%A7%E5%85%AC%E6%9C%89%E9%9B%B2-ai-%E5%B9%B3%E5%8F%B0)
- 必要性的業務優化需求：千萬不要為了 AI 而 AI，而是要有專業人員先參與評估可以優化那些內部業務需求，如果不知道，也不想找人評估，那行政單據OCR應該會是最容易讓高層看到效果的試金石了。
- 低薪資但高回報的人力：全台都在喊需要AI人才，特別讓人懷念所謂的資安即國安的口號；無奈，不說所謂只給的出香蕉來請猴子，現在是怕給了也找不到猴子啊。


-----

儘管如前所述，台灣企業在導入 AI 的過程中，確實存在高層心態、成本投入 與人才 等多重挑戰；然而，這並不代表所有企業都只是停滯不前高大上；在金融、電信、製造與半導體龍頭，已陸續發布了具體的 GenAI 落地產品或內部流程優化案例可以參考。

### 台灣上市櫃公司 GenAI 實際落地案例總表 (2024-2025)

| 產業類別 | 公司名稱 | 落地應用 / 平台名稱 | 具體流程優化 (實際案例) | 開發模式 (推測) |
| :--- | :--- | :--- | :--- | :--- |
| **金融業** | 玉山金控 (E.Sun FHC) | 內部 "GENIE" 平台 | • 輔助 IT 撰寫程式碼<br>• 輔助法遵比對法規文件<br>• 輔助業務撰寫 KYC 報告 | **In-House 主導** (串接 Azure OpenAI) |
| **金融業** | 國泰金控 (Cathay FHC) | 內部 "Copilot" 平台 | • 輔助 IT 程式碼生成與除錯<br>• 自動生成會議記錄摘要<br>• 升級智能客服 "阿發" | **In-House 主導** (串接 Azure OpenAI) |
| **金融業** | 富邦金控 (Fubon FHC) | 導入保險核心業務 | • 【核保】自動摘要體檢報告<br>• 【理賠】自動分析診斷書與單據 | **In-House / 混合** |
| **電信業** | 中華電信 (Chunghwa) | "DeepVoice" 分析平台 | • 客服全量通話進行情緒與意圖分析<br>• AI 輔助 5G 基地台選址 | **In-House / 混合** |
| **電信業** | 台灣大哥大 (TWM) | 智能客服 "小麥" | • 支援更複雜的多輪對話<br>• 提供個性化行銷與產品推薦 | **In-House + PaaS** |
| **電信業** | 遠傳電信 (Far EasTone) | 內部 "FriGPT" 平台 | • 輔助 IT 程式碼生成<br>• 輔助法務審核合約<br>• 內部規章SOP知識庫 | **In-House 主導** |
| **科技業** | 聯發科 (MediaTek) | 晶片產品 (如 T930) | • 將 GenAI 功能落地到終端 (Edge AI)<br>• (非內部流程，而是核心產品) | **產品整合** |
| **科技業** | 台達電 (Delta) | 研發 (R\&D) 知識庫 | • 工程師用自然語言查詢技術文檔<br>• 加速研發創新與除錯 | **In-House 主導** |
| **科技業** | 華碩 (ASUS) | 內部知識平台 | • 內部 AI 知識庫<br>• IT 營運維護助手 | **混合 (共同開發)** (與 Microsoft 合作) |
| **智慧製造** | 鴻海 (Foxconn) | "AI 工廠" | • 學習老師傅經驗 (如注塑)<br>• 數位孿生與物流模擬 | **In-House / 混合** |
| **智慧製造** | 廣達電腦 (Quanta) | 智慧工廠 | • AI 伺服器產線的生產排程調度<br>• 預測良率異常，提出改善建議 | **混合 (共同開發)** (與 NVIDIA 等合作) |
| **智慧製造** | 研華 (Advantech) | "AI 廠務助理" (WISE-PaaS) | • 廠務主管用「自然語言」查詢 OEE<br>• 自動撈數據並用白話文回報 | **PaaS / 產品整合** |
| **半導體** | 台積電 (TSMC) | 研發 (R\&D) 與製造 (Fab) | • 輔助先進製程的電路佈局 (Layout) 優化<br>• 自動生成 Fab 的「良率異常摘要報告」 | **In-House (高度機密)** |
| **半導體** | 日月光 (ASE) | 智慧工廠 (封測) | • 從 AI 瑕疵檢測，升級到「根本原因分析」<br>• 自動生成報告說明瑕疵成因 | **混合 (In-House + 共同開發)** |
| **零售電商** | 統一超商 (7-ELEVEN) | OPEN POINT 會員經營 | • 自動生成「千人千面」的行銷文案<br>• 升級智能客服，處理複雜點數問題 | **PaaS + 顧問導入** |
| **零售電商** | PChome 網路家庭 | 平台優化 | • 優化站內「語意搜尋」的理解力<br>• 自動化生成「商品文案」 | **雲平台 (PaaS)** (與 Google Cloud 合作) |
| **零售電商** | 富邦媒體 (momo) | 平台優化 | • "AI 虛擬主播" (直播帶貨)<br>• "AI 智慧選品" (輔助生成文案) | **混合 (In-House + PaaS)** |
| **傳統產業** | 台塑集團 (FPG) | 營運與合規 | • 輔助撰寫 ESG 報告書草稿<br>• 工安事件分析與通報建議 | **In-House** (透過台塑網) |
| **運輸服務** | 長榮/華航 | 智能客服 / 營運 | • AI 客服處理複雜票務 (改票、差價)<br>• 輔助分析師「生成」動態定價建議 | **顧問/SI 導入** |

-----

## 導論

企業導入 AI 的旅程，就如同在充滿潛在回報但也遍布陷阱的挑戰。多數專案的沒有然，並非因為 AI 技術本身，而是因為某個關鍵階段迷失了方向。本本將根據自身體驗以及相關文章提供的指南，供一份實踐路線圖，提供企業從「如何開始」出發，走過戰略、基礎建設、人員治理等關鍵階段，並特別標示出在每個階段最容易犯下的錯誤，以及如何應對生成式 AI（GenAI）帶來的全新挑戰。

## 序：為何是現在？
- 解鎖AI：從最基礎的概念出發，釐清人工智慧、機器學習、深度學習到當前最熱門的生成式AI之間的關係與區別。
- 洞見AI：跨越不同行業，展示AI如何在金融、行銷、營運、醫療等領域解決真實的商業問題，創造可衡量的價值。
- 擘劃AI：提供一個策略框架，探討企業在導入AI時必須考量的組織、數據、人才與風險管理等關鍵議題，並提供具體的起步建議。

## 第一章：解鎖AI

### 建立一套清晰的AI詞彙與概念體系
- 通用人工智慧 (Artificial General Intelligence, AGI)
- 狹義人工智慧 (Artificial Narrow Intelligence, ANI)

#### 機器學習：預測的藝術
垃圾郵件過濾器：傳統的作法是工程師手動編寫數百條規則（例如，「如果郵件標題包含『免費』、『中獎』就歸為垃圾郵件」），但這種方法僵化且容易失效；機器學習的做法？
- 機器學習的主要功能是進行預測與分類。它擅長回答以下兩類問題：
    - 分類問題：「這位客戶會流失嗎？」（是/否）、「這筆信用卡交易是詐欺嗎？」（是/否）。
    - 預測（迴歸）問題：「我們下個季度的銷售額會是多少？」（一個具體的數值）、「這棟房子的合理售價是多少？」（一個具體的數值）。
值得注意的是，傳統的機器學習模型最擅長處理的是結構化數據，也就是那些可以整齊地存放在試算表或資料庫中的數據，例如銷售數字、客戶基本資料、庫存量等。

#### 深度學習：讓數據「開口說話」
- 深度學習(DL)是機器學習的一個更強大、更先進的子集。它並非一項全新的技術，而是機器學習技術在處理特定問題上的一種演進與深化。幼兒學習辨識貓的過程。我們不會對一個孩子說：「貓有毛、有鬍鬚、有尖耳朵」。相反地，我們只是不斷地指給他看各種各樣的貓——黑貓、白貓、坐著的貓、奔跑的貓。透過觀察大量範例，孩子的大腦會自動形成一個複雜的、多層次的概念模型來辨識「貓」。
- 深度學習的超能力在於其處理非結構化數據的卓越表現。非結構化數據是指那些無法輕易放入傳統資料庫的數據，例如圖片、影片、聲音、自然語言文本等。

#### 生成式AI：創造的黎明
- 生成式AI (Generative AI) 是AI領域的一項典範轉移。與我們前面討論的、主要用於分析或分類現有數據的AI不同，生成式AI的核心能力是創造全新的、原創的內容。這些內容可以是文字、圖像、程式碼、音樂，甚至是影片。
- 在商業上，生成式AI的主要功能是創造、摘要與互動。它能夠回應這樣的指令：「為我的新產品草擬一封行銷郵件」、「將這份長篇報告總結成500字的摘要」，或者「扮演一個專業的客服人員，回答客戶問題」。它正從一個分析工具，轉變為一個知識工作者的強大協作夥伴。

### 為戰略設定正確的航向與規劃
這是所有 AI 專案計畫的起點，也是最容易犯下根本性錯誤的地方。方向錯了，再努力也是枉然，更別說是跟著想曝光的高層瞎搞了。

#### 如何開始：
AI 計畫的起點**絕對不是「我們需要 AI」**，應該是 **「我們有一個棘手的業務問題」**。

#### 致命錯誤 1：戰略脫節
**陷阱**：在沒有明確業務目標的情況下，僅因 **「競爭對手在做」** 或 **「高層一時興起」** 而啟動專案。
- 這類專案從一開始就缺乏存在的理由，最終因無法證明其商業價值而被放棄。
- 最常碰到的便是新聞都在播AI，高層也很常聽到AI，所以就希望大家用力 **拍腦袋** 想想看能夠怎樣 AI；但這通常正是最會然後沒有然後的案例。

#### 致命錯誤 2：期望與現實的鴻溝
**陷阱**：將 AI 視為能立即解決所有問題的「魔法」，卻嚴重 **「低估了背後所需的數據準備、基礎設施和人才投入」**。
- 如同 [Bernard Marr 在 The AI Graveyard: 7 Deadly Mistakes That Kill Most Enterprise AI Projects](https://bernardmarr.com/the-ai-graveyard-7-deadly-mistakes-that-kill-most-enterprise-ai-projects/) 所提到：一家公司曾因其數據分散在 27 個老舊系統中，導致耗資 250 萬美元的 AI 專案徹底失敗，這就是期望與現實脫節的慘痛代價。
- 但在台灣，估計不太可能有公司高層突然的投入250萬美元 (7000多萬新台幣) 要做AI，所以這點通常是還沒發生或者只是發個新聞稿，然後就沒有然後了 !

#### 正確航線：
- 首先應識別出一個具體、可衡量且對公司至關重要的業務挑戰（例如：如何將業務處理時間縮短降低 20%）。
- 採取漸進式方法。從一個小規模、能快速看到成效的試點專案開始，驗證可行性、累積經驗並建立內部信心，然後再逐步擴大。
- [看看 Meta 和 Open AI等公司為了做AI在人力與硬體上投入了多少](https://iknow.stpi.niar.org.tw/Post/Read.aspx?PostID=21985) ?
- 至少找一個 **「相對有實戰及學術經驗的人」(EX: 至少有資訊領域碩博士學歷，或已在資訊科技領域工作多年)**，不要想著便宜好辦事，或者找壓根不是資訊背景或隨便上個幾堂課的網紅 。


## 第二章：打造堅實基礎的後勤基地

如果說 AI 是前線的作戰部隊，那數據和基礎設施就是決定戰爭勝敗的後勤。

### 如何開始：
在啟動 AI 專案前，先誠實地評估自身的數據狀況。企業應先具備穩定的數據倉儲、商業智慧（BI）和基礎分析能力。

### 致命錯誤 3：數據困境（垃圾進，垃圾出）
**陷阱**：這是 AI 專案失敗最常見的技術原因。組織往往假設現有數據「足夠好」，卻忽略了數據品質不佳、格式不一、標籤混亂等問題。
- 一個醫療系統曾試圖用 AI 預測病患再入院率，最終卻因各設施的數據編碼不一致，導致模型學到的是「數據的混亂」而非「醫療的規律」。

### 致命錯誤 4：跳過基礎工作
**陷阱**：在連基礎的銷售報告都無法統一的情況下，就想直接打造先進的 AI 定價系統。
- 這如同在沙上蓋樓，註定會崩塌。AI 是數據成熟度的演進，而非技術上的蛙跳。


## 第三章：擘劃您的AI藍圖：組建團隊並制定從策略到實踐的規則

再好的技術和數據，如果沒有合適的人來使用和管理，也只是一堆昂貴的擺設。

### 如何開始：
建立一個跨職能的 AI 團隊或卓越中心，並從專案第一天起就讓最終使用者（例如工廠主管、行銷人員）真心願意參與進來；畢竟，這事不在他們原本所謂的 KPI 上，80%的人估計都不願意另外花時間來協助正站在風口上的所謂 AI 團隊。

### 致命錯誤 5：缺乏人性化因素
**陷阱**：將 AI 專案視為純粹的 IT 任務，完全忽略使用者的感受和工作流程。
- 一樣參考 [Bernard Marr 在 The AI Graveyard: 7 Deadly Mistakes That Kill Most Enterprise AI Projects](https://bernardmarr.com/the-ai-graveyard-7-deadly-mistakes-that-kill-most-enterprise-ai-projects/) 所提到：一個耗資 180 萬美元的製造業 AI 系統，就因工廠主管們不信任、不理解而完全不使用其建議，最終宣告失敗。
- 不過，這裡是台灣，180萬美元相當於5000多萬台幣，估計不會有這麼衝動的高層。

### 致命錯誤 6：人才與治理不足
**陷阱**：缺乏兼具技術與商業頭腦的人才，同時又沒有明確的治理架構。這會導致各部門各自為政、重複投資。
- 還是 [Bernard Marr 在 The AI Graveyard: 7 Deadly Mistakes That Kill Most Enterprise AI Projects](https://bernardmarr.com/the-ai-graveyard-7-deadly-mistakes-that-kill-most-enterprise-ai-projects/) 所提到：一家電信公司就曾因七個部門各自開發 AI，最終浪費數百萬美元，專案也被迫取消。
- 同上，這裡是台灣，這卻是非常可能發生的場景，與對岸所謂敵對國家的內卷相比較，台灣較常見的反倒是高層各擁山頭，誰也不讓誰，但也誰都做不出啥鬼東西。


## 第四章：應對 GenAI — Navigating the New Minefield

GenAI 帶來了巨大的機遇，但也埋下了全新的、更隱蔽的地雷。管理這些風險，是當下最重要的課題。

### 如何開始：

立即制定一份清晰的 GenAI 使用政策。這是所有後續風險管理的基石，也是一個「不費吹灰之力」卻至關重要的決定。若無政策，就等於默許所有風險的發生。

### 致命錯誤 7：取代人類創造力，引發真實性危機
**陷阱**：過度依賴 GenAI 產出通用、缺乏靈魂的內容，會讓品牌顯得廉價。
- 遊戲發行商動視暴雪（Activision Blizzard）就因使用「AI 垃圾」取代人類創作而引發粉絲強烈反彈。GenAI 應是增強創意的工具，而非替代品。

### 致命錯誤 8：盲目信任產出，忽略人為監督
**陷阱**：GenAI 會犯錯，高達 46% 的 AI 生成文本可能包含事實錯誤。
- 科技媒體 CNET 就因其 AI 生成的報導錯誤百出而嚴重損害聲譽。任何關鍵產出都必須有「人在迴路中」進行審核。

### 致命錯誤 9：未能保護機密資料
**陷阱**：員工可能在不知情下，將公司機密或客戶個資輸入 ChatGPT 等公共工具，造成數據外洩。
- 三星員工的案例已為全球企業敲響警鐘。

### 致命錯誤 10：忽視智慧財產權風險
**陷阱**：在商業上使用由可能包含版權資料訓練的 GenAI 所產出的內容，可能讓企業在未來面臨侵權訴訟的風險。

---

## 結論：成功的 AI 遠征需要一張好地圖

企業導入 AI 的成功，取決於是否能在每個階段都做出正確的決策。這份路線圖與其中的錯誤警示，就是幫助您規劃路徑、避開陷阱的地圖。謹慎、有策略且治理良好的方法，是通往可持續成功的唯一途徑。



<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/AIBeginner"
  },
  "headline": "企業 AI新賽局 入門策略實踐路線圖：從策略到治理，避開致命陷阱",
  "description": "一份提供給企業的 AI 入門策略與實踐路線圖，內容涵蓋從 AI 概念釐清、戰略規劃、數據基礎建設到團隊治理的關鍵階段，並指出各階段常見的致命錯誤與應對 GenAI 挑戰的建議。",
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
  "datePublished": "2025-07-21",
  "dateModified": "2025-07-21",
  "keywords": "Enterprise AI, AI Strategy, GenAI, Machine Learning, Deep Learning, AI Adoption, 企業 AI, 人工智慧導入, 數位轉型"
}
</script>