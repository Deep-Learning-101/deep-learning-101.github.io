---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---


{% include header.html %}

---

{% include ai-share.html %}

---

# 臺灣大型語言模型及文字嵌入和重排序模型性能評測與在地化策略分析報告
_Llama-3.1-Taiwan-8B-Instruct、Llama-3.1-Taiwan-8B、Llama-3-Taiwan-8B-Instruct-128k、Llama-3-Taiwan-8B-Instruct-DPO、Llama-3-Taiwan-8B-Instruct、Llama-3-Taiwan-70B-Instruct-128k、Llama-3.1-TAIDE-LX-8B-Chat、Llama-Breeze2-3B-Instruct、Llama-Breeze2-8B-Instruct、gemini-embedding-001、Qwen3-Embedding、Qwen3-Reranker_

**作者**：[TonTon Huang Ph.D.](https://www.twman.org/)  
**日期**：2025年07月16日更新

**相關文章參考**：
* <b><a href="https://blog.twman.org/2024/09/LLM.html" target="_blank">大型語言模型直接就打完收工？</a></b>：<a href="https://deep-learning-101.github.io/1010LLM">回顧 LLM 領域探索歷程，討論硬體升級對 AI 開發的重要性。</a>
* <b><a href="https://blog.twman.org/2024/07/RAG.html" target="_blank">檢索增強生成(RAG)不是萬靈丹之優化挑戰技巧</a></b>：<a href="https://deep-learning-101.github.io/RAG">探討 RAG 技術應用與挑戰，提供實用經驗分享和工具建議。</a>
* <b><a href="https://blog.twman.org/2024/02/LLM.html" target="_blank">大型語言模型 (LLM) 入門完整指南：原理、應用與未來</a></b>：<a href="https://deep-learning-101.github.io/0204LLM">探討多種 LLM 工具的應用與挑戰，強調硬體資源的重要性。</a>
* <b><a href="https://blog.twman.org/2023/04/GPT.html" target="_blank">解析探索大型語言模型：模型發展歷史、訓練及微調技術的 VRAM 估算</a></b>：<a href="https://deep-learning-101.github.io/GPU">探討 LLM 的發展與應用，硬體資源在開發中的作用。</a>
* <b><a href="https://www.facebook.com/cnanewstaiwan/posts/pfbid02CCrFhyvCcoTmjJaX4aHaSMHmCgnPd1SG21Gbpb4Wo9bgs7QmQArTmhbVPZSLyjrdl" target="_blank">中央社繁體中文預訓練資料集案</a></b>

---

### **文章目錄**

- [引言](#引言)
- [✨LLM-API-Platform-Price-Comparison](#llm-api-platform-price-comparison)
  - [總體戰略比較三大公有雲-ai-平台](#總體戰略比較三大公有雲-ai-平台)
  - [自行架設-gpu-vm-每小時預估費用](#自行架設-gpu-vm-每小時預估費用)
  - [大型語言模型api平台價格比較-202507](#大型語言模型api平台價格比較-202507)
  - [大型語言模型與agent安全工具比較-202507](#大型語言模型與agent安全工具比較-202507)
- [臺灣大型語言模型性能評測與在地化策略分析](#臺灣大型語言模型性能評測與在地化策略分析)
  - [i-目的與核心發現](#i-目的與核心發現)
  - [ii-繁體中文大型語言模型評測基準深度解析](#ii-繁體中文大型語言模型評測基準深度解析)
  - [iii-模型綜合性能評測分析](#iii-模型綜合性能評測分析)
  - [iv-核心洞察與策略意涵](#iv-核心洞察與策略意涵)
  - [v-建議與展望](#v-建議與展望)
  - [vi-結論與洞見](#vi-結論與洞見)
- [從零到一：打造本地端高精準度 RAG 系統的實戰指南 (涵蓋Embedding-與-Reranking-模型在-RAG-應用中的關鍵角色與評估)](https://deep-learning-101.github.io/RAG)


---

## 引言

隨著人工智慧技術的快速發展，專為臺灣本土文化與語言環境優化的大型語言模型（Large Language Models，LLMs）逐漸嶄露頭角。為了客觀評估這些模型的能力，研究人員建立了多種基準測試（benchmarks），特別針對繁體中文及臺灣特定知識領域的理解能力進行測試。本文將匯整臺灣本土模型以及國際知名模型（如Qwen和Llama系列）在這些基準測試上的表現結果。

---

### **最新重要模型發布與技術突破 ~2025年08月05日**

2025年08月，AI領域迎來了三家頂尖公司的重要模型發布，標誌著大型語言模型和世界模型技術的新突破。這些發布不僅展現了技術進步，也反映了不同公司在AI發展路線上的策略差異。

#### **[OpenAI 的 gpt-oss 開源系列](https://openai.com/zh-Hant/index/introducing-gpt-oss/)**

The gpt-oss models, including the 120b version rivaling o4-mini in reasoning benchmarks on a single 80GB GPU and the 20b version matching o3-mini with just 16GB memory, enable efficient deployment for edge computing and local inference applications.

OpenAI 首次發布開源模型系列，打破了其一貫的閉源策略：[OpenAI重新開源！深夜連發兩個推理模型，o4-mini水平](https://www.jiqizhixin.com/articles/2025-08-06-2)

| 模型 | 參數規模 | 核心特點 | 硬體需求 | 性能比較 | 技術挑戰 |
|------|----------|----------|----------|----------|----------|
| **[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)** | 120B | 媲美 o4-mini 推理能力 | 單張 80GB GPU | Andrew Ng：「looks strong」 | 幻覺率 49%，程式設計不穩定 |
| **gpt-oss-20b** | 20B | 接近 o3-mini 性能 | 僅需 16GB 記憶體 | 邊緣運算友好 | 幻覺率 53%，但整體表現穩定 |

**技術評估**：
- **優勢**：硬體效率突出，為本地部署和邊緣運算開啟新可能
- **挑戰**：在 PersonQA 內部基準測試中，兩個模型的幻覺率（49% 和 53%）都顯著高於 OpenAI 的閉源模型 o1（16%）和 o4-mini（36%）
- **實用性**：gpt-oss-20b 在實際應用中反而比 120b 版本更穩定，特別適合程式設計任務

#### **Google DeepMind 的 Genie 3 世界模型**

[震撼，世界模型第一次超真實地模擬了真實世界：GoogleGenie 3昨晚搶了OpenAI風頭](https://www.jiqizhixin.com/articles/2025-08-06-5)

Genie 3 is an advanced world model capable of building coherent, interactive worlds over extended periods, simulating physical properties, natural ecosystems, animations, novels, and diverse locations or historical settings through breakthroughs in controllable real-time interactions.

Genie 3 代表了世界模型技術的重大突破，具備構建完整、連貫且可互動世界的能力：

**核心功能**：
- **物理模擬**：精確展現水流、閃電等自然現象及複雜環境互動
- **生態系統建模**：生成動態生態環境，包含動物行為和植物生命週期
- **創意內容生成**：創造奇幻場景和富有表現力的動畫角色
- **時空穿越**：跨越地理和歷史界限，模擬不同時代場景

**技術突破**：
- **長期記憶一致性**：環境在數分鐘內保持高度物理一致性
- **即時互動響應**：每秒多次計算以回應用戶輸入
- **時間軌跡追蹤**：能回溯並引用一分鐘前的相關資訊
- **自回歸穩定性**：克服誤差累積問題，維持長期連貫性

#### **Anthropic 的 Claude Opus 4.1**

[阻擊OpenAI，Claude搶先數十分鐘發布Claude Opus 4.1](https://www.jiqizhixin.com/articles/2025-08-06-4)

Claude Opus 4.1 offers hybrid reasoning modes for instant responses or visible processes, with fine-tuned thinking budgets for cost-efficiency, excelling in advanced programming on SWE-bench and long-term agent-based research across diverse data sources.

Claude Opus 4.1 在推理能力和實用性方面取得顯著進展：

**核心特色**：
- **混合推理模式**：支援即時回應和詳細推理過程展示
- **思維預算控制**：API 用戶可精細調整成本與性能平衡
- **超大輸出支援**：32K token 輸出能力

**領域優勢**：

| 應用領域 | 具體表現 | 技術優勢 |
|----------|----------|----------|
| **高階程式設計** | SWE-bench 基準領先 | 完成耗時數日的工程任務，支援大規模程式碼重構 |
| **智能體研究** | 長達數小時自主研究 | 綜合分析專利資料庫、學術論文、市場報告 |
| **編碼穩定性** | 「出乎意料地穩定」 | 在編碼能力對比中表現最強且最穩定 |

#### **跨模型比較與產業影響**

**編碼能力對比**：
根據實際測試，Claude Opus 4.1 在編碼能力方面表現最佳且穩定性突出。相較之下，gpt-oss-120b 在程式設計任務中表現不穩定，而 gpt-oss-20b 反而展現更好的整體穩定性。

**效率與智慧平衡**：
Artificial Analysis 測試顯示，gpt-oss 系列雖在純智慧水準上落後於 DeepSeek R1 和 Qwen3 235B，但在運算效率方面具有明顯優勢，體現了「參數規模與性能效率」的新平衡點。

**產業策略意涵**：
- **OpenAI**：首次開源策略標誌著競爭格局的重大變化
- **Google**：專注世界模型技術，開拓全新應用領域
- **Anthropic**：強化推理能力和實用性，鞏固企業應用市場

---

<p align="center">
  🎵 不聽可惜的 NotebookLM <audio controls style="width:200px; height:20px;"><source src="../notebooklm-mp3/TW-LLM-Benchmark.mp3" type="audio/mpeg"></audio> Podcast @ Google 🎵
</p>

---

<div style="display: flex; justify-content: center;">
  <div style="position: relative; width: 100%; max-width: 450px; aspect-ratio: 16 / 9;">
    <iframe
      src="https://www.youtube.com/embed/V2lxdga6N1Y"
      style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowfullscreen>
    </iframe>
  </div>
    <div style="position: relative; width: 100%; max-width: 450px; aspect-ratio: 16 / 9;">
    <iframe
      src="https://www.youtube.com/embed/fiQCsMJ_hdg"
      style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowfullscreen>
    </iframe>
  </div>
</div>

---

{% include price.html %}

---


# 臺灣大型語言模型性能評測與在地化策略分析

## I. 目的與核心發現

當前的 LLM 市場呈現出一個多元且競爭激烈的格局。一方面，全球開源巨頭（如 Meta 的 Llama 系列、阿里巴巴的 Qwen 系列）以及頂尖閉源模型（如 OpenAI 的 GPT-4 系列、Anthropic 的 Claude 系列）的激烈競爭，在通用知識與推理能力上設立了極高的性能標竿。另一方面，由臺灣本地團隊（如 TAME 計畫、國家級的 TAIDE 計畫、以及科技巨頭聯發科的 MediaTek-Research）所推動的在地化開源模型，正積極建立其在特定領域的領導地位；這些模型致力於彌補全球模型在繁體中文語境和臺灣文化理解上的不足，針對臺灣特定文化、法律及語境的評測項目上，展現出顯著且可量化的「在地化優勢」（Localization Premium）。此優勢在處理高度在地化的任務時尤為突出，證明了在地化微調的不可替代性。
在眾多模型中，yentinglin/Llama-3-Taiwan-70B-Instruct-DPO 在臺灣專屬評測項目中表現最為亮眼，成為處理在地化任務的首選。與此同時，Qwen/Qwen2-72B-Instruct 則在通用學科知識評測（如 MMLU 和 TMLU）中展現出卓越的實力，其強大的基礎能力使其在某些臺灣學術型評測中甚至超越了在地化模型，這一現象揭示了模型規模、預訓練資料品質與在地化微調之間複雜的交互關係。

## II. 繁體中文大型語言模型評測基準深度解析
要客觀評估大型語言模型的真實能力，必須先深入理解所使用的評測基準（Benchmark）。每一個評測基準都有其獨特的設計哲學、評估維度與內在限制。因此需詳細解析評測所涵蓋的各項關鍵基準，為後續的性能數據分析提供必要的背景知識與批判性視角。

### A. 評測基準的重要性與挑戰
大型語言模型評測基準是標準化的測試集，旨在衡量和比較不同語言模型在各種任務上的能力，例如語言理解、問答、數學解題和程式編寫等。透過在公開的基準上進行測試，研究人員和開發者可以相對客觀地比較不同模型的性能，並在排行榜上展示其成果。
然而，標準化評測也面臨諸多挑戰。其中最主要的是「資料污染」（Data Contamination）問題，即模型在預訓練階段可能已經「看過」評測集中的題目與答案，導致評測分數虛高，無法反映其真實的泛化能力。此外，許多評測基準的焦點較為狹隘，可能無法全面評估模型的綜合能力。隨著模型技術的飛速發展，現有評測基準也可能迅速飽和或失去挑戰性，無法有效地區分頂尖模型的優劣。因此，在解讀評測分數時，必須對這些潛在限制保持警覺。

### B. 國際標準評測：MMLU
MMLU（Massive Multitask Language Understanding，大規模多任務語言理解）是一項廣泛使用的英文基準測試，評估模型在多種英語任務上的表現；目前頂尖模型如GPT-4在MMLU上的成績達到86.4%[1]。
* 定義：MMLU 旨在衡量模型在預訓練過程中獲取的廣泛知識。它包含 57 個學科的選擇題，涵蓋 STEM（科學、技術、工程和數學）、人文學科、社會科學等領域，難度從初級教育延伸至專業級別。其評估方式主要為零樣本（zero-shot）和少樣本（few-shot）學習，這種設置更接近人類接受評量的模式，也更具挑戰性。
* 重要性：由於其廣泛的學科覆蓋和高難度，MMLU 已成為衡量模型通用知識和解決問題能力的黃金標準。它提供了一個關鍵的參照點，讓我們能夠將臺灣的在地化模型與國際頂尖模型（如 GPT-4, Llama 3, Claude 3）置於同一尺度下進行比較。
* 限制與演進：儘管 MMLU 應用廣泛，但研究指出其存在一些問題，例如部分題目存在事實錯誤、選項模糊或有多個正確答案等，這可能導致模型得分的上限並非 100%。同時，資料污染的風險也持續威脅其有效性。為應對 MMLU 逐漸飽和的趨勢，學術界已開發出更具挑戰性的後繼版本，如 MMLU-Pro。MMLU-Pro 透過增加選項數量（從 4 個增加到 10 個）和引入更需要複雜推理的題目，顯著提升了評測的難度與穩定性，這也反映了全球評測基準為跟上模型發展而持續演進的趨勢。

### C. 臺灣特化評測：TMLU
TMLU是一個專門針對臺灣繁體中文環境設計的綜合性基準測試，涵蓋國中、高中、大學及國家考試等多個教育和專業領域的知識評估[3][4]。
* 定義：TMLU 包含 37 個學科，範圍橫跨國中、高中至專業級別，內容涵蓋社會科學、STEM、人文學科以及臺灣特有主題。其題目形式為多選題，總計約 3,000 題 [3][4]。
* 設計哲學：TMLU 最核心的設計理念在於「對抗資料污染」。為了最大限度地降低模型在訓練時接觸過評測題目的風險，TMLU 的出題來源主要為網路上的 PDF 和 Word 文件，而非直接從網頁抓取的純文字。這一點與其他直接從單一網站抓取題目的評測（如 TMMLU+）形成鮮明對比。此外，TMLU 還為每個學科人工撰寫了少量「思維鏈」（Chain-of-Thought, CoT）的範例，以引導和評估模型的複雜推理能力。
* 評論與觀點：儘管 TMLU 在設計上力求嚴謹，但仍有評論指出其不足之處。例如，有審稿人認為，部分 STEM 領域的題目用詞仍偏向中國大陸的術語，且整體題目設計未能充分捕捉臺灣華語獨特的語言學與文化特徵，這或許可以解釋為何一些針對簡體中文優化的模型在 TMLU 上也能取得優異成績。

### D. 臺灣特化評測：TMMLU+
TMMLU+是TMLU的增強版本，擁有更全面的臺灣繁體中文評估內容[5][6]。
* 定義：TMMLU+ 是一個包含 22,690 道選擇題的龐大資料集，涵蓋從國小到專業級別的 66 個學科。相較於前代版本，TMMLU+ 的規模擴大了六倍，並致力於實現更均衡的學科分佈。該評測集明確包含了臺灣特有的文化主題，如臺灣法律、農業實務、原住民文化等。
* 與 TMLU 的關係：在臺灣的 AI 社群中，TMLU 與 TMMLU+ 形成了兩種不同設計哲學的代表。TMLU 將「控制資料污染」置於首位，追求評測的純淨性與穩健性；而 TMMLU+ 則優先考慮「規模」與「學科廣度」，力求評測的全面性。這種良性競爭反映了臺灣 LLM 評測生態系的活力，但也意味著開發者在評估模型時，可能需要在兩種不同的「真相來源」之間做出選擇。
* 評論與觀點：針對 TMMLU+ 的主要批評集中在其資料來源。由於其大部分題目可追溯至單一的線上題庫網站，這大大增加了資料污染的風險。此外，有分析指出，在 TMMLU+ 上使用 CoT 提示策略反而會降低多數模型的性能，這與在 MMLU-Pro 等強調推理的評測上的表現截然相反，暗示 TMMLU+ 所測試的可能更偏向於知識檢索而非複雜推理。

### E. 專項能力評測

除了綜合知識評測外，針對特定能力的專項評測也至關重要。

* **TW Truthful QA**
此評測旨在評估模型在臺灣特定背景下回答問題的「真實性」（truthfulness），即模型生成準確資訊並避免傳播錯誤觀念的能力 [7]。其方法論源自國際上通用的 TruthfulQA 基準，該基準的核心理念是設計一些容易引導人類產生錯誤信念的問題，來測試模型是否會模仿這些常見的謬誤。評估方式通常依賴一個經過微調的「裁判模型」（GPT-judge）來判斷生成答案的真實性與資訊量。因此，TW Truthful QA 的分數可以視為模型「在地化真實性」的一項指標。此基準測試專門評估模型以臺灣特定背景回答問題的能力，測試模型對臺灣文化、社會和歷史等本土知識的掌握程度，以及在地化能力[7][8][9]。
* **TW Legal Eval**
這是一項高難度的專業領域評測，其題目直接來源於臺灣的律師資格考試 [7]。此評測專門衡量模型對臺灣法律術語、法學概念和法律推理的掌握程度。在全球範圍內，法律領域的 AI 應用都被視為高風險、高標準的場景，對準確性有著極端嚴苛的要求。因此，模型在 TW Legal Eval 上的表現，是其是否具備專業級應用潛力的重要參考指標。使用臺灣律師資格考試的問題來評估模型對臺灣法律術語和概念的理解能力[3][7][9]。這項測試直接檢驗模型處理臺灣特有法律體系和專業知識的能力[9][10]。


## III. 模型綜合性能評測分析

本章節將呈現本次研究的核心成果：對一系列臺灣在地化及國際主流大型語言模型在關鍵評測基準上的綜合性能數據，並進行深入的比較與剖析。透過將分散於各處的評測數據匯總於一處，我們得以進行橫向與縱向的對比，從而揭示不同模型之間的細微差異及其背後的策略意涵。為了更精準地評估頂尖模型的推理能力，本節的數據總表特別新增了更具挑戰性的 MMLU-Pro 評測項目。

### A. 關鍵評測數據總表

為了提供一個清晰、全面的比較視角，下表匯總了本次研究涵蓋的主要模型在多個關鍵評測基準上的表現。數據主要來源於公開的 Open TW LLM Leaderboard [7][11]、各模型的官方發布文件及相關學術論文。所有分數均以百分比（%）表示，除非另有說明。標示為「N/A」表示目前尚無公開的可靠數據。


[**臺灣本土與主流大型語言模型綜合評測結果（更新版**](https://huggingface.co/datasets/ikala/tmmluplus#benchmark-on-direct-prompting)


| 模型名稱 | TMLU | TMMLU+ | TW Truthful QA | TW Legal Eval | TW MT-Bench | MMLU (5-shot) | MMLU-Pro (CoT, 5-shot) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **yentinglin/Llama-3-Taiwan-70B-Instruct** | 74.76% | 67.53% | 80.95% | 68.42%| 7.54| N/A | N/A |
| **yentinglin/Llama-3-Taiwan-70B-Instruct-DPO** | 74.60%| - | 81.75%| 70.33%| - | N/A | N/A |
| **yentinglin/Llama-3-Taiwan-70B-Instruct-128k** | 73.01%| - | 80.16%| 63.64%| - | N/A | N/A |
| **yentinglin/Llama-3-Taiwan-8B-Instruct** | 59.50%| 52.28%| 61.11%| 53.11%| 7.21| N/A | N/A |
| **yentinglin/Llama-3-Taiwan-8B-Instruct-DPO** | 59.88%| - | 59.52%| 52.63%| - | N/A | N/A |
| **yentinglin/Llama-3-Taiwan-8B-Instruct-128k** | - | - | - | - | - | N/A | N/A |
| **taide/Llama-3.1-TAIDE-LX-8B-Chat (Alpha1)**| 47.30%| 39.03%| 50.79%| 37.80%| - | N/A | N/A |
| **Breeze-7B-Instruct-v1_0**| 55.57%| 41.77%| 52.38%| 39.23%| 6.0| N/A | N/A |
| **Breexe-8x7B-Instruct-v0_1**| - | 48.92%| - | - | 7.2| N/A | N/A |
| **MediaTek-Research/Llama-Breeze2-8B-Instruct** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **MediaTek-Research/Llama-Breeze2-3B-Instruct** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Qwen/Qwen3-235B-A22B** | N/A | N/A | N/A | N/A | N/A | 87.8%| N/A |
| **Qwen/Qwen2.5-72B-Instruct** | N/A | N/A | N/A | N/A | N/A | 86.1%| 71.6%|
| **Qwen/Qwen2-72B-Instruct** | 80.43%| N/A | 73.02% | 58.85% | N/A | 82.30%| N/A |
| **Qwen1.5-110B-Chat**| 75.69%| 65.81%| 66.67%| 49.28%| - | N/A | N/A |
| **meta-llama/Llama-4-Maverick** | N/A | N/A | N/A | N/A | N/A | 85.5% | 80.5% |
| **meta-llama/Llama-4-Scout** | N/A | N/A | N/A | N/A | N/A | 79.6% | 74.3% |
| **meta-llama/Llama-3.3-70B-Instruct** | N/A | N/A | N/A | N/A | N/A | N/A | 68.9% |
| **meta-llama/Llama-3.2-11B-Vision** | N/A | N/A | N/A | N/A | N/A | 46.4% | N/A |
| **meta-llama/Llama-3.2-3B-Instruct** | N/A | N/A | N/A | N/A | N/A | 63.4% | N/A |
| **meta-llama/Llama-3.1-70B-Instruct** | N/A | N/A | N/A | N/A | N/A | 83.6% | 66.4% |
| **meta-llama/Llama-3.1-8B-Instruct** | N/A | N/A | N/A | N/A | N/A | 69.4% | 48.3% |
| **meta-llama/Llama-3-70B-Instruct** | 70.95%| 62.75%| 65.08%| 52.63%| - | 82.0% | N/A |
| **meta-llama/Llama-3-8B-Instruct** | 55.81%| 43.38%| 46.83%| 35.89%| - | 68.4% | N/A |
| **openai/GPT-4o** | 69.88% (5-shot)| N/A | 76.98%| 53.59%| - | 88.70% | 77.9% |
| **anthropic/Claude-3-Opus** | 73.59% (5-shot)| 64.20% | 69.84%| 60.29%| - | 86.80% | N/A |

這張總表的建立本身即具備高度價值。它將散落在不同排行榜、論文和模型發布頁面上的數據整合在一起，首次提供了一個全面的、可直接進行並排比較的單一視圖。正是基於這張表，我們才能夠進行後續的深入分析，例如量化在地化微調的具體效益，或比較不同規模模型之間的性能差距。

### B. 臺灣特化模型深度剖析
#### 1. yentinglin/Llama-3-Taiwan 系列
Llama-3-Taiwan系列是由[林彥廷（Yenting Lin)](https://yentingl.com/)開發的模型，基於Meta的Llama-3架構，專為繁體中文使用者進行微調[7][9][19]。這些模型在臺灣本土基準測試上表現優異，尤其是70B參數版本[7][9]。

Llama-3-Taiwan-70B-Instruct在TMLU測試中達到74.76%的成績，Taiwan Truthful QA達到80.95%，Legal Eval達到68.42%[7][9]。尤其值得注意的是，其DPO版本在Taiwan Truthful QA和Legal Eval測試中取得了最佳成績，分別為81.75%和70.33%[7][9]。

8B參數版本雖然參數量較小，但仍達到了令人印象深刻的表現，TMLU為59.50%，Taiwan Truthful QA為61.11%，Legal Eval為53.11%[7][9]。這表明即使在較小的參數規模下，經過良好的本土化微調，模型也能取得不錯的理解能力[8][9][19]。

* **性能分析**：從上表可見，70B 參數級別的模型在各項臺灣專屬評測中均名列前茅。特別是 yentinglin/Llama-3-Taiwan-70B-Instruct-DPO 版本，在講求事實準確性的 TW Truthful QA（81.75%）和專業知識密集的 TW Legal Eval（70.33%）上均拔得頭籌 [9]。這表明，在標準指令微調（Instruct）之後，額外採用直接偏好優化（Direct Preference Optimization, DPO）技術，能有效提升模型在特定價值觀（如真實性）和專業領域（如法律）上的對齊程度與推理能力。
* **不同版本的權衡**：在 8B 級別中，Instruct、DPO 和 128k 三種版本各有側重。標準的 Instruct 版本在各項在地化評測中表現均衡 [9]。DPO 版本在通用知識 TMLU 上略高於標準版（59.88% vs 59.50%），但在 TW Truthful QA 上略低（59.52% vs 61.11%）[9]，顯示出不同優化技術帶來的細微性能取捨。而 128k 長文本版本目前則缺乏公開的標準評測數據 [7][9]。

#### 2. taide/Llama-3.1-TAIDE-LX-8B-Chat
TAIDE（Trustworthy AI Dialogue Engine）是由臺灣國家科學及技術委員會推動的計畫，其推出的模型具有官方背景，備受關注。所開發的Llama3-TAIDE-LX-8B-Chat-Alpha1模型是另一個專為臺灣本土環境優化的模型[20]。根據數據，該模型在TMLU上達到47.30%，Taiwan Truthful QA為50.79%，Legal Eval為37.80%，TMMLU+為39.03%[7]。

TAIDE模型特別強化了繁體中文處理能力，並針對長文本理解進行優化，從8K擴展到131K的上下文長度[20]。但是與其它以臺灣本土相關LLM系列相比還有差距，尚需深入優化改善，期許不要又跟往常一樣只是雷聲大雨點小的計畫[7][20]。
* **性能分析**：最新發布的 Llama-3.1-TAIDE-LX-8B-Chat 基於 Meta 最新的 Llama-3.1 模型進行開發，使其擁有比 yentinglin/Llama-3-Taiwan-8B（基於 Llama-3）更先進的架構基礎 [20][21]。然而，從 Open TW LLM Leaderboard 的數據來看，其早期 Alpha 版本 (Llama3-TAIDE-LX-8B-Chat-Alpha1) 的綜合表現落後於 yentinglin 的 8B 模型 [7]。TAIDE 團隊在其官方發布中，更側重於評估模型在辦公室常用任務（如摘要、翻譯、信件撰寫）上的表現，並在這些任務上展示了相較於 Llama-3.1-8B-Instruct 基礎模型的進步。但在更具挑戰性的長文本評測 CLongEval 上，其表現則不如基礎模型，這可能與其訓練資料和策略側重於特定任務有關 [20]。

#### 3. MediaTek-Research/Llama-Breeze2 系列
由臺灣半導體巨頭聯發科（MediaTek）研究院推出的 Breeze2 系列，代表了臺灣產業界在大型語言模型領域的頂尖實力。包括Breeze-7B-Instruct-v1_0和Breexe-8x7B-Instruct-v0_1[22][23][24]。Breeze-7B-Instruct-v1_0在TMLU上達到55.57%，Taiwan Truthful QA為52.38%，Legal Eval為39.23%，TW MT-Bench為6.0，TMMLU+為41.77%[7][24]。
最新的Breeze2系列基於Llama 3.2架構，專為繁體中文進行優化，提供3B和8B兩種參數規模，除了語言理解能力外，還整合了視覺能力和函數呼叫功能[24][25]。Breeze2系列專為處理繁體中文知識進行了優化，包含台灣特定的文化和社會背景[24]。

* **技術特點**：Breeze2 系列建立在非常新的基礎模型之上，其 8B 版本基於 Llama 3.1，而 3B 版本更是基於 Llama 3.2 [24][26]。這意味著它們從一開始就繼承了 Meta 最先進的模型架構和能力。更重要的是，Breeze2 系列從設計之初就整合了視覺理解（vision-aware）和函數呼叫（function-calling）等多模態能力，使其應用場景遠超純文字模型 [24][27]。
* **數據缺口**：儘管 Breeze2 系列在技術上極具潛力，但一個關鍵問題是，截至本報告撰寫之時，尚未有公開的、在 TMLU, TMMLU+, 或 MMLU 等標準評測上的成績 [24]。雖然其較早期的版本（如 Breeze-7B-Instruct-v1_0）曾在一些排行榜上出現，但這些數據無法代表最新模型的真實水平 [7]。這個數據缺口使得我們無法將最新的 Breeze2 模型與其他模型進行直接的量化比較，這是目前臺灣 LLM 評測生態系中的一個顯著遺憾。

### C. 與國際及中國大陸主流模型之比較分析
#### 1. Llama 3 & 4 (Meta)
Meta 的 Llama 系列是全球開源模型的標竿，其官方公布的 MMLU 成績是我們評估模型通用知識能力的基準線。從 Llama 3 到 Llama 4，我們可以看到清晰的性能演進路徑。Llama-3.1-70B-Instruct 在 MMLU 上獲得 83.6% 的高分，而在更難的 MMLU-Pro 上得分為 66.4% [14]。最新的 Llama-4-Maverick 則在 MMLU-Pro 上達到了 80.5%，展現了架構和訓練方法上的巨大進步 [14]。這些分數代表了未經特定在地化微調的「全球基準」水平。

Meta的Llama原版模型也在臺灣本土基準測試中有所表現[7][9]。Meta-Llama-3-70B-Instruct在TMLU上達到70.95%，Taiwan Truthful QA為65.08%，Legal Eval為52.63%，TMMLU+為62.75%[7][9]。

Meta-Llama-3-8B-Instruct的表現則相對較弱，TMLU為55.81%，Taiwan Truthful QA為46.83%，Legal Eval為35.89%，TMMLU+為43.38%[7][9]。這與臺灣本土優化的yentinglin/Llama-3-Taiwan-8B-Instruct（TMLU 59.50%、TW Truthful QA 61.11%、Legal Eval 53.11%、TMMLU+ 52.28%）相比，明顯有差距[7][9]。

最新的Llama 3.3系列繼續改進了多語言能力，但主要集中在英語、德語、西語、葡語、義語、法語、泰文和北印度語（Hindi），尚無專門針對繁體中文的優化[15][21]。在MMLU等基準測試上，Llama 3.3達到86.0分（0-shot, CoT），與Llama 3.1 70B持平[15]，但在更具挑戰性的MMLU PRO（5-shot, CoT）上達到68.9分，優於Llama 3.1 70B[15]。

#### 2. Qwen 2 & 3 (Alibaba)
阿里巴巴的 Qwen 系列是另一個極具競爭力的開源模型家族。Qwen2-72B-Instruct 不僅在 MMLU 上取得了 82.3% 的高分 [13]，更在臺灣的 TMLU 評測上取得了 80.43% 的驚人成績 [7][13]。其後繼者 Qwen2.5-72B-Instruct 在 MMLU 上提升至 86.1% [12]，而最新的 Qwen3-235B 更是達到了 87.8% [12]，持續刷新開源模型的性能上限。

Qwen（通義千問）系列模型，尤其是Qwen1.5-110B-Chat，在臺灣本土基準測試上表現出色[9][18]。在TMLU測試中，Qwen1.5-110B-Chat達到75.69%，是所有測試模型中的最高分[7][9][13]。在Taiwan Truthful QA上達到66.67%，Legal Eval為49.28%，TMMLU+為65.81%[7][9][13]。

Qwen2和Qwen2.5系列進一步提升了多語言能力，在MMLU等國際基準測試上表現優異[18]。例如，Qwen2.5-72B在MMLU上達到86.1%，MMLU-Pro達到71.6%[12][18]。不過，目前尚無這些新版本在臺灣本土基準測試上的完整數據[18]。

#### 3. 可量化的「在地化優勢」
透過直接比較在地化模型與其基礎模型，我們可以精確地量化在地化微調所帶來的價值。以 yentinglin/Llama-3-Taiwan-8B-Instruct 與其基礎模型 meta-llama/Llama-3-8B-Instruct 為例，數據（源自 [7], [9]）顯示：
* 在 TMLU 上：從 55.81% 提升至 59.50%，增加 3.69 個百分點。
* 在 TMMLU+ 上：從 43.38% 提升至 52.28%，增加 8.90 個百分點。
* 在 TW Truthful QA 上：從 46.83% 提升至 61.11%，增加 14.28 個百分點。
* 在 TW Legal Eval 上：從 35.89% 提升至 53.11%，增加 17.22 個百分點。

這一系列的數據提供了確鑿的證據：在地化微調並非錦上添花，而是能夠帶來實質性、大幅度性能提升的關鍵步驟。特別是當任務的文化、語言及專業領域的在地化程度越高時（如法律、事實性問答），這種性能提升就越加顯著。
然而，Qwen2-72B 在 TMLU 上的卓越表現（80.43%）[7][13]，甚至超越了頂尖的在地化模型 yentinglin/Llama-3-Taiwan-70B（74.76%）[9]，構成了一個值得深思的現象。這挑戰了「在地化模型永遠是最佳選擇」的簡單論述。
這種看似矛盾的結果背後，可能有多重原因。首先，TMLU 作為一個學術知識導向的評測，其內容（特別是 STEM 領域）在不同語言文化間具有高度的普適性，這部分題目更多地考驗模型的基礎推理與知識儲備，而非文化細節。一個像 Qwen2 這樣經過海量、高品質、多樣化資料（即使以簡體中文為主）預訓練的超大規模模型，可能已經發展出極其強大的底層通用推理能力，使其能夠在這些普適性問題上表現出色 [28]。其次，這也反過來印證了在地化微調的真正價值所在。在更需要文化細膩度和在地語境的評測上，例如 TW Truthful QA，yentinglin 的模型依然保持著明顯的領先優勢（80.95% vs. 73.02%）[7][9][13]。
結論是，在地化的價值並非一成不變，而是與任務特性緊密相關。對於涉及普適性知識的任務，模型的基礎規模與訓練品質可能佔據主導地位；而對於深度嵌入在地文化、語言習慣和專業領域（如法律、在地時事）的應用，經過精準微調的在地化模型則擁有不可替代的優勢。這為開發者在不同應用場景下選擇模型提供了更為細緻的決策依據。

## IV. 核心洞察與策略意涵

在前一章節的數據分析基礎上，本章節旨在提煉出更深層次的洞察，並探討其對於臺灣 AI 產業發展的策略性意涵。這些洞察涵蓋了在地化的價值、模型規模的權衡，以及臺灣在當前全球開源生態系中所面臨的機遇與挑戰。

### A. 在地化的絕對價值：何時與為何重要
綜合第三章的數據分析，尤其是在地化模型與其基礎模型的直接比較，我們可以得出一個明確的結論：在地化微調具有絕對且不可或缺的價值。這種價值並非均勻分佈在所有任務上，而是在特定場景下表現得尤為突出。
當應用場景高度依賴對臺灣的社會文化、時事動態、專業領域知識（如法律、醫療）以及獨特語言習慣的精準理解時，在地化模型不僅是「更好」的選擇，甚至是「唯一」可行的選擇。yentinglin/Llama-3-Taiwan 系列在 TW Legal Eval 和 TW Truthful QA 這兩項評測上相較於其 Llama-3 基礎模型所展現出的巨大性能差距（分別高出 17.22 和 14.28 個百分點），便是最有力的證明 [7][9]。
這背後的邏輯在於，全球模型（無論是來自美國還是中國大陸）的預訓練資料中，關於臺灣的內容佔比極低，導致它們在面對臺灣特有的概念、實體和事件時，容易產生「事實幻覺」（hallucination）或給出模糊、不確定的答案。在地化微調通過注入大量高品質的在地化資料，有效地彌補了這一「知識盲區」，從而顯著提升了模型在這些關鍵應用中的可靠性與準確性。因此，對於金融、法律、政府公共服務、在地化客服等高風險或高價值的應用，投資於在地化模型是確保服務品質與使用者信任的必要策略。

### B. 模型規模的權衡：性能與成本的博弈
模型參數的規模是影響其性能的核心變數之一，但更大的模型也意味著更高的運算成本。我們的分析揭示了規模與性能之間的非線性關係，為開發者在性能與成本之間進行權衡提供了依據。
比較 yentinglin 系列的 8B 和 70B 模型，在 TMLU 評測上，後者比前者高出約 15 個百分點（74.76% vs. 59.50%）[9]。同樣，比較 Meta 官方的 Llama 3.1 8B 和 70B Instruct 模型，在 MMLU 上的差距也達到了 14.2 個百分點（83.6% vs. 69.4%）[14]。
這些數據表明，從 80 億（8B）參數規模躍升至 700 億（70B）參數規模，所帶來的性能提升是顯著的、階梯式的，而非微不足道的邊際改善。70B 級別的模型在知識的廣度、推理的深度和處理複雜指令的能力上，都遠超 8B 級別的模型。
然而，這種性能的飛躍伴隨著巨大的運算成本。70B 模型的推理（inference）對硬體資源的需求遠高於 8B 模型，這直接影響到應用的部署成本和回應延遲。這就形成了一個經典的權衡困境：
* **70B+ 模型**：代表了當前的性能巔峰，適用於對準確度和複雜度要求極高的旗艦級應用或高價值商業場景。
* **~8B 模型**：代表了一個「物有所值」的性能門檻，其能力足以勝任大量中等複雜度的任務，同時部署成本更低、速度更快，是許多新創公司和中小型應用的理想選擇。

對於開發者而言，這意味著不存在一個普適的最佳選擇。決策的關鍵在於精準評估應用場景對模型能力的需求，並在可接受的成本範圍內，選擇能夠滿足該需求的最小、最高效的模型。

### C. 開源生態系的現況：臺灣的機遇與挑戰
臺灣在大型語言模型領域的發展，展現出一個充滿活力且日漸成熟的開源生態系，這既是機遇，也伴隨著挑戰。
* **機遇**：臺灣擁有如 yentinglin/Llama-3-Taiwan、taide/Llama-3.1-TAIDE-LX、MediaTek-Research/Llama-Breeze2 等多個由學術界、政府及產業界頂尖團隊推動的在地化開源模型項目 [7][20][24]。這一方面證明了臺灣具備從模型微調、資料處理到評測建構的完整技術實力；另一方面，也為臺灣的企業和開發者提供了豐富的選擇，使其不必完全依賴於國外的模型，從而降低了技術自主性的風險，並能更好地滿足在地化需求。
* **挑戰**：儘管在地化模型在特定領域表現出色，但 Qwen2 在 TMLU 上的強勢表現敲響了警鐘 [7][13]。它揭示了一個嚴峻的現實：來自全球的超大規模模型，憑藉其龐大的預訓練資料和雄厚的運算資源，正在建立起極高的通用能力壁壘。臺灣的在地化模型若想保持競爭力，必須持續深化其差異化優勢。這包括：
    * 更深度的在地化：不僅是語言，更要深入文化、價值觀和特定行業的知識圖譜。
    * 更快的技術迭代：需要緊跟全球基礎模型的發展步伐，在最新的模型（如 Llama 3.1, Llama 3.2）發布後，迅速進行在地化微調。
    * 更完善的生態協作：如 Breeze2 最新模型評測數據的缺失所反映的，臺灣的開源生態系需要在標準化、即時性及透明化的公開評測上做得更好，以利於整個社群的協同發展和良性競爭。

總而言之，臺灣的開源 LLM 生態系正處於一個關鍵的發展階段。未來的成功將取決於能否在利用全球先進基礎模型的同時，憑藉獨特的在地化數據和領域知識，打造出在全球競爭格局中具有明確比較優勢的產品。

## V. 建議與展望

基於前述詳盡的數據分析與洞察，本章節將提供具體的模型選型建議，並對未來可能的研究方向與產業趨勢進行展望，以期為臺灣 AI 領域的開發者、研究人員及決策者提供前瞻性的參考。

### A. 開發者選型建議
選擇合適的大型語言模型是成功開發 AI 應用的第一步。以下是針對不同應用需求的具體建議：
* **追求極致臺灣在地化性能**：
    * **推薦模型**：yentinglin/Llama-3-Taiwan-70B-Instruct-DPO
    * **理由**：此模型在 TW Legal Eval 和 TW Truthful QA 等高度考驗在地知識與事實準確性的評測中表現最佳 [9]。對於需要處理臺灣法律文件、提供在地化資訊查詢、或進行深度文化內容生成的應用，該模型是當前開源領域的首選。其 DPO 微調進一步強化了模型的可靠性。
* **尋求成本與性能的最佳平衡點**：
    * **推薦模型**：yentinglin/Llama-3-Taiwan-8B-Instruct
    * **理由**：在 8B 參數規模下，此模型在所有臺灣專屬評測項目上均顯著優於其 Llama-3 基礎模型，提供了扎實的在地化能力，同時保持了相對較低的部署成本和較快的推理速度 [7][9]。對於預算有限、但仍需可靠在地化表現的應用（如一般性聊天機器人、內容草稿生成），這是一個極具性價比的選擇。
* **需要頂尖通用推理與多語言能力**：
    * **推薦模型**：Qwen/Qwen3-235B-A22B 或 meta-llama/Llama-4-Maverick
    * **理由**：這些模型在 MMLU 和 MMLU-Pro 等國際通用評測上均展現出世界級的水平，證明其擁有極強的基礎推理能力 [12][14]。如果您的應用場景不完全侷限於深度臺灣文化，而是需要處理跨學科的複雜問題、或涉及多語言內容，這些模型是非常強大的選項。
* **基於最新架構進行二次開發**：
    * **推薦模型**：taide/Llama-3.1-TAIDE-LX-8B-Chat 或 MediaTek-Research/Llama-Breeze2-8B-Instruct
    * **理由**：這兩款模型均基於 Meta 最新的 Llama 3.1 或 Llama 3.2 架構開發，繼承了其長文本、多語言和工具使用等先進特性 [20][21][24]。雖然它們目前的公開評測數據尚不完整或不如其他模型亮眼，但其現代化的基礎架構使其成為進行進一步領域微調（domain-specific fine-tuning）的絕佳起點。

### B. 未來研究方向
當前的評測結果也為未來的學術研究指明了幾個關鍵方向：
* **下一代評測基準的開發**：TMLU 和 TMMLU+ 的設計與其受到的批評，共同揭示了開發更穩健、更能抵抗資料污染、且能深入評估模型文化細膩度的下一代評測基準的迫切性。未來的評測應更側重於檢測模型在面對臺灣特有俚語、雙關語、歷史典故和價值觀衝突時的表現，這些是僅靠模型規模難以克服的挑戰。
* **解析並超越「Qwen 現象」**：Qwen 系列在 TMLU 上的優異表現值得深入研究。未來的研究應致力於釐清，究竟是其龐大的預訓練資料、更優的訓練演算法、還是其他因素，使其通用能力能夠遷移至臺灣的學術評測場景。同時，臺灣的研究社群也應探索如何透過更高效的在地化微調策略，在所有類型的任務上（而不僅僅是深度文化任務）都建立起對全球模型的明確優勢。
* **臺灣情境下的多模態評測**：隨著 Breeze2 等多模態模型的出現，臺灣的 LLM 發展已進入文生圖、圖生文的新階段 [24][27]。然而，相應的在地化多模態評測仍處於起步階段，如 VisTai 等基準尚在發展中。建立一套全面、涵蓋臺灣在地視覺元素（如街景、美食、文化地標）的綜合性多模態評測基準，將是推動下一波 AI 創新的關鍵基礎設施。

### C. 產業展望
展望未來，臺灣的 LLM 產業將呈現「多模型共存」的格局。企業將不再尋求一個「萬能模型」，而是會根據不同業務需求，採用「多 LLM 策略」（multi-LLM strategy），為特定任務選擇最適合的工具。例如，法律部門可能採用在 TW Legal Eval 上表現最佳的模型，而行銷部門則可能選擇在創意寫作上更具優勢的模型。
在地化模型與全球模型的競爭將持續加劇，這將成為推動臺灣模型品質提升和成本下降的主要動力。臺灣 AI 生態系的長期成功，將取決於能否充分利用獨特、高品質的在地化資料（包括文字、圖像與聲音），並在金融、醫療、製造、法律等臺灣具有優勢的垂直領域進行深度耕耘，從而打造出全球模型難以複製的專業護城河。這場競賽不僅是技術的較量，更是數據、領域知識與生態系協作能力的綜合比拼。

## VI. 結論與洞見

從收集的資料可以得出以下幾點洞見：

1. **臺灣本土優化的模型在臺灣特定知識上表現優越**：經過臺灣本土資料微調的模型，如Llama-3-Taiwan系列，在Taiwan Truthful QA和Legal Eval等臺灣特定知識測試上，表現優於原版的國際模型[7][9]。這凸顯了本土化微調對提升模型在特定文化和語言環境下理解能力的重要性[3][8][9]。
2. **大型參數模型普遍表現更佳**：70B參數級別的模型通常優於8B級別的模型，但經過良好微調的小型模型也能達到不錯的效果[7][9]。例如，yentinglin/Llama-3-Taiwan-8B-Instruct在多項指標上優於原版Meta-Llama-3-8B-Instruct[9]。
3. **國際模型的強項與弱項**：Qwen系列等國際模型在通用知識上表現優異，甚至在某些臺灣本土測試上也取得了良好成績[9][13]。然而，在深度結合臺灣文化、法律等特定領域知識的測試上，本土優化模型通常更勝一籌[7][9]。
4. **繼續改進的空間**：即使是表現最好的模型，在某些測試上仍有提升空間[3][7][9]。特別是在法律評估（Legal Eval）上，最高分也僅為70.33%（yentinglin/Llama-3-Taiwan-70B-Instruct-DPO）[9]。
5. **本土化與通用能力的平衡**：理想的模型應當在保持強大通用能力的同時，具備優秀的本土化理解能力[3][8][19]。臺灣本土模型的持續發展將致力於在這兩方面取得更好的平衡[7][19]。

總體而言，臺灣本土大型語言模型在繁體中文和臺灣特定知識的理解上展現出了顯著優勢，證明了針對特定語言和文化背景進行模型優化的價值[3][7][9]。同時，國際頂尖模型也在不斷提升多語言能力，這種良性競爭將推動大型語言模型技術的整體進步[1][18][21]。


**參考文獻**

[1]: Galileo.ai Blog. *MMLU Benchmark: Everything you need to know*. (對應 MMLU 介紹)

[2]: https://arxiv.org/abs/2502.16428. MMLU 相關論文

[3]: arXiv:2403.20180v1 *TMLU: A Benchmark for Traditional Chinese Language Understanding in Taiwan*. (TMLU 主要論文)

[4]: https://arxiv.org/abs/2403.20180. TMLU 相關，可能與[3]重複，確認是否需要兩個

[5]: Hugging Face. *lianghsun/Llama-3.2-Taiwan-3B*. (TMMLU+ 相關模型)

[6]: LinkedIn post by Liang-Hsun Huang. *lianghsun/Llama-3.2-Taiwan-3B-Instruct*. (TMMLU+ 及 Llama-3.2-Taiwan MMLU 分數)

[7]: GitHub. *MiuLab/Taiwan-LLM*. (Open TW LLM Leaderboard 及臺灣特化評測數據來源)

[8]: PromptLayer. *Llama-3-Taiwan-8B-Instruct Model Card*.

[9]: Hugging Face. *yentinglin/Llama-3-Taiwan-70B-Instruct* (及其系列模型頁面，包含各項臺灣評測數據).

[10]: Taipei Times. *Academics develop AI model for legal queries*. (TW Legal Eval 相關新聞)

[11]: Hugging Face Spaces. *yentinglin/open-tw-llm-leaderboard*. (Open TW LLM Leaderboard)

[12]: QwenLM Blog. *Qwen2.5: A new generation of large language models*. (Qwen2.5, Qwen3 MMLU/MMLU-Pro 分數)

[13]: Hugging Face. *Qwen/Qwen2-72B*. (Qwen2-72B 官方數據及 TMLU 分數)

[14]: arXiv:2407.07796. *Llama 4 Series*. (Llama 4 and Llama 3.1 MMLU/MMLU-Pro scores)

[15]: DataCamp Blog. *Llama 3.3 70B: Meta’s Latest Open LLM*. (Llama 3.3 MMLU-Pro 分數)

[16]: https://arxiv.org/html/2403.01858v1. 可能指 Llama 3 早期報告或 MMLU 5-shot Leaderboard 相關

[17]: Semantic Scholar. Paper 0a39c40528b4604bfb7ef9b18c354af2e7eb4366. (GPT-4o MMLU-Pro 數據可能來源)

[18]: ITHome News. *Anthropic釋出Claude 3.5 Sonnet，比肩GPT-4o、Gemini 1.5 Pro*. (Claude-3-Opus TMMLU+ 及 Qwen 系列新聞)

[19]: NTU CSIE Announcement. *Yenting Lin - Taiwan LLM Development, Evaluation, and Insights*.

[20]: TAIDE Official Website. *TAIDE Project*. (TAIDE 計畫官方網站及 Llama-3.1-TAIDE-LX-8B-Chat 資訊)

[21]: arXiv:2403.01858v3. *The Llama 3 Series of Models*. (Llama 3.1 報告)

[22]: MediaTek Tek Talk Blogs. *MediaTek Research Breeze 7B: A Highly Efficient LLM*.

[23]: https://arxiv.org/html/2403.02712v1. Breeze 相關論文

[24]: MediaTek Taiwan Blog. *聯發創新基地全面開源 Breeze2*. (Breeze2 官方發布及資訊)

[25]: https://arxiv.org/html/2501.13921v1. Breeze2 相關論文

