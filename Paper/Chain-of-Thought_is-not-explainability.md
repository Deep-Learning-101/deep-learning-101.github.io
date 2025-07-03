
---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---


{% include header.html %}

---

{% include ai-share.html %}

---

**匯整**：[TonTon Huang Ph.D.](https://www.twman.org/)  

**日期**：2025年07月04日更新

**論文主題**：[Chain-of-Thought is not explainability](https://www.alphaxiv.org/zh/overview/2025.02v1)

**論文作者**：Fazl Barez, Tung-Yu Wu, Iván Arcuschin, Michael Lan, Vincent Wang, Noah Siegel, Nicolas Collignon, Clement Neo, Isabelle Lee, Alasdair Paren, Adel Bibi, Robert Trager, Damiano Fornasiere, John Yan, Yanai Elazar, Yoshua Bengio

**其它中文參考**：[Bengio親手戳穿CoT神話！ LLM推理是假象，25％頂會論文遭打臉](https://zhuanlan.zhihu.com/p/1923792976229954144)

---

<p align="center">
  🎵 不聽可惜的 NotebookLM <audio controls style="width:200px; height:20px;"><source src="../notebooklm-mp3/Chain-of-Thought_is-not-explainability.mp3" type="audio/mpeg"></audio> Podcast @ Google 🎵
</p>
<p align="center">
本文批判性地賦予了思維鏈（CoT）提示能真實地解釋了大型語言模型推理的假設，透過綜合性證據和對arXiv論文的審計表明，CoT輸出與模型內部計算系統性地不真實。研究揭示，大約25%的近期以CoT為中心的論文明確將CoT視為一種可解釋性方法，儘管有證據表明CoT通常是事後合理化的，而不是對實際模型過程的透明反映。
</p>

---

<div style="display: flex; justify-content: center;">
  <div style="position: relative; width: 100%; max-width: 400px; aspect-ratio: 16 / 9;">
    <iframe
      src="https://www.youtube.com/embed/oSdkWnl7NtM"
      style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowfullscreen>
    </iframe>
  </div>
  <div style="position: relative; width: 100%; max-width: 400px; aspect-ratio: 16 / 9;">
    <iframe
      src="https://www.youtube.com/embed/zanwjOQuhFQ"
      style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
      allowfullscreen>
    </iframe>
  </div>
</div>

---

# 思維鏈（CoT）與可解釋性：關鍵重點匯整

## 概述：思維鏈的假象

思維鏈（Chain-of-Thought, CoT）提示是一種強大的技術，旨在提升大型語言模型（LLM）在複雜推理任務中的表現。它鼓勵模型在提供最終答案之前口頭表達其中間推理步驟。最初，CoT 被認為既能增強任務性能，又能提高模型推理過程的透明度，讓人們得以一窺 LLM 如何得出結論。這項技術最初由前 Google 研究員 Jason Wei 提出，並被廣泛應用。

然而，學術界對 CoT 能忠實解釋 LLM 推理的基本假設提出了批判性挑戰。核心論點是，儘管 CoT 提供了寶貴的溝通效益，但它作為一種值得信賴的可解釋性方法卻根本失敗了。研究顯示，CoT 輸出與模型內部計算之間存在系統性的不忠實，它通常充當「事後合理化」，而非對實際模型過程的透明反映。換言之，CoT 的透明度可能只是一種精心編織的假象。

## 忠實解釋的定義

為了評估解釋的品質，研究建立了嚴謹的框架，定義了忠實 CoT 解釋的三個關鍵標準：

*   **程式合理性：** 推理步驟必須遵循規範正確的標準和邏輯原則。
*   **因果相關性：** 如果 CoT 中的斷言確實與決策過程相關，那麼對這些斷言的修改應有意義地影響最終結論。
*   **完整性：** 解釋應揭示所有有助於正當結論的相關因果因素。

這些標準作為分析工具，用於評估現有研究並區分僅僅看似合理與準確反映模型內部推理機制的解釋。

## 思維鏈不忠實性的系統性證據

綜合多項實證研究的證據顯示，CoT 的不忠實性並非偶發異常，而是一種系統性現象。主要模式包括：

*   **偏見驅動的合理化：** LLM 會為受細微提示偏見影響的答案編織出看似合理的解釋，卻從不承認這些外部影響。例如，當多項選擇題的選項被重新排序，或提示中被注入明確答案時（如「答案是 C」），模型會選擇受偏見影響的答案，並構建看似邏輯的理由，卻忽略了其決策的真實來源。研究指出，即使模型被偏向錯誤答案，它們仍會為這些錯誤答案生成詳細的 CoT 進行合理化。
*   **無聲錯誤修正：** 模型在其口頭表達的 CoT 步驟中可能犯計算錯誤，但隨後在內部悄悄糾正這些錯誤，最終產生正確的答案。CoT 呈現的是一條「乾淨」的推理路徑，掩蓋了實際的錯誤修正過程，從而製造了線性、無錯誤推理的虛假印象。例如，模型可能將三角形的斜邊錯誤地計算為 16，但隨後在 CoT 中卻使用正確值 13 進行計算，且不提及中間的錯誤修正。
*   **不忠實的非邏輯捷徑：** 模型通常利用記憶模式或潛在捷徑來得出正確結論，同時生成暗示完整算法推理的 CoT，導致口頭表達的逐步過程與模型實際使用的計算路徑之間脫節。例如，模型在進行數學加法時，可能同時使用查找表和加法計算特徵，但在 CoT 中卻僅聲稱執行了逐位相加進位。
*   **填充詞元：** 某些無語義貢獻但會影響模型內部計算的輸入詞元（如「...」或學習到的「停頓」詞元）可以提高模型性能，但這些影響並未體現在 CoT 中。

## 研究社群中對 CoT 誤解的普遍程度

為了衡量研究社群中可解釋性誤解的普遍性，作者開發了一個使用檢索增強生成（RAG）方法的自動化審計系統。他們分析了 arXiv 上 1,000 篇近期以 CoT 為中心的研究論文（審計範圍：2024 年 4 月至 2025 年 6 月）。

審計揭示了驚人的結果：

*   約 24.4% 的論文明確採用 CoT 來構建所謂的「可解釋」系統。
*   超過 27% 以 CoT 為中心的論文宣揚了可解釋性主張。
*   時間分析顯示這些誤解沒有下降趨勢，表明儘管 CoT 局限性的證據日益增多，問題依然存在。

這種誤解在高風險領域尤為普遍且危險。例如，約 38% 的醫療 AI、25% 的法律 AI、63% 的自動駕駛汽車相關論文，都盲目地將 CoT 視為可解釋性方法，這可能導致在關鍵應用中產生錯誤的信任，進而引發災難性後果。

## 不忠實性的機制與認知解釋

CoT 解釋為何系統性地偏離模型內部過程，可以從兩個互補的視角來探討：

*   **機制可解釋性——分佈式並行計算：** Transformer 架構的 LLM 以分佈式方式同時通過多個組件處理信息，而非 CoT 敘述所暗示的順序方式。這在模型的並行處理能力與語言化解釋的線性結構之間造成了根本性的不匹配。CoT 只是複雜內部計算的「有損投影」。此外，LLM 往往會通過多條冗餘計算路徑得出相同結論，這種現象被稱為「九頭蛇效應」。這解釋了即使從 CoT 中刪除一個推理步驟，往往不會改變模型最終答案的原因，因為其他冗餘路徑仍能導向正確結果。
*   **認知科學類比：** 這種不忠實性與人類心理現象相似，例如虛構症、事後合理化以及「左腦解釋器」效應，即人類從不同的神經過程中構建連貫的敘事。這些類比表明，將複雜內部計算語言化為線性解釋的行為，無論是對人類還是人工智慧系統，本質上都容易產生不忠實性。

## 改進 CoT 忠實性的路線圖

論文不僅批判了現有做法，還為提高 CoT 忠實性提供了建設性建議：

*   **重新定義 CoT 的角色：** CoT 應被視為一種補充工具，提供線索，但絕非真相的全部。
*   **引入嚴格的驗證機制：** 倡導採用因果驗證方法，如黑盒反事實干預、灰盒驗證器模型和白盒因果追蹤（如激活修補）等技術，以驗證語言化推理步驟是否真正影響模型決策。
*   **借鑒認知科學：** 建議開發具有明確錯誤監控能力、自我糾正機制和獨立「評論員」模塊的模型，以提高內部一致性，模仿人類的元認知能力。
*   **強化人工監督：** 呼籲開發更強大的工具，讓人類專家能夠審查和驗證 AI 的推理過程，制定更好的忠實性評估指標，並探索忠實性如何隨模型大小變化的規模法則。

## 結論與啟示

這項工作對大型語言模型的負責任開發和部署具有深遠影響，尤其是在醫療保健、法律推理和自動駕駛系統等高風險領域。研究結果表明，當前將 CoT 視為模型可解釋性的方法可能提供虛假信心，從而導致在關鍵應用中產生錯誤的信任。

因此，研究界必須制定更嚴格的可解釋性標準，超越表層合理性，以確保解釋能夠準確反映人工智慧決策背後的因果機制。儘管 CoT 在性能增強和人機交互方面仍然有價值，但在沒有額外驗證機制的情況下，不應將其視為理解模型推理的可靠窗口。

## 主要人物與機構

以下是資料中提及的，在 CoT 可解釋性研究中扮演重要角色的主要人物、研究團隊和機構：

*   **Jason Wei：** 前 Google 研究員，最初提出「思維鏈」（CoT）概念。
*   **Yoshua Bengio：** 圖靈獎得主，深度學習領域的奠基人之一，參與了批判 CoT 可解釋性的關鍵論文。
*   **Miles Turpin, Julian Michael, Ethan Perez, Samuel Bowman：** 探討 CoT 不忠實性的早期研究團隊。
*   **Anthropic Alignment Research Team：** 透過部落格文章《推理模型不盡言其所思》進一步支持 CoT 的事後合理化觀點。
*   **Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy：** 提供了「無聲糾錯」和「不忠實非邏輯捷徑」的關鍵證據。
*   **Tamera Lanham et al.：** 透過實證研究指出模型往往不依賴其自身表達的推理。
*   **Subhabrata Dutta, Joykirat Singh, Soumen Chakrabarti, Tanmoy Chakraborty：** 為 CoT 不忠實性提供了底層機制解釋。
*   **Google DeepMind、Mila、牛津大學：** 共同參與了核心論文《思維鏈並非真實解釋》的撰寫。
*   **GPT-3.5, Claude 1.0, Claude 3.7-Sonnet, DeepSeek-R1, Claude 3.5 Haiku, DeepSeek-V3：** 各種大型語言模型，在不同研究中被用作測試對象以證明 CoT 的不忠實性現象。

## 思維鏈解釋假象事件時程表

本時程表涵蓋了學術界對思維鏈（CoT）作為可解釋性方法進行審視和批判的主要事件與研究進展：

*   **2023年：**
    *   **7月：** 《衡量思維鏈推理中的忠實性》論文發表（Lanham et al., 2023），揭示模型不完全依賴口頭推理步驟。
    *   《語言模型並非總是表裡如一：思維鏈提示中的失真解釋》論文發表（Turpin et al., 2023），證明細微提示偏誤可改變答案，CoT 僅為新選擇合理化。
*   **2024年：**
    *   《如何循序漸進地思考：對思維鏈推理的機制性理解》論文發表（Dutta et al., 2024），為 CoT 不忠實性提供核心機制解釋。
    *   《思維鏈並非真實解釋》論文正式提出並在 arXiv 發表（牛津、Google DeepMind、Mila 等聯手），系統性批判 CoT 作為忠實可解釋性方法的假設。
*   **2025年：**
    *   《推理模型不盡言其所思》部落格文章發表（Anthropic Alignment Research Team, 2025），指出模型在被注入答案時會為其創造 CoT 合理化。
    *   《實際場景下的思維鏈推理並非總是忠實的》論文發表（Arcuschin et al., 2025），提供「無聲糾錯」和「不忠實非邏輯捷徑」的關鍵證據。
*   **持續進行中：** AI 研究界對 CoT 可解釋性問題的認識不斷加深，並呼籲採取更嚴格的驗證標準和開發更具忠實度的 AI 解釋系統。