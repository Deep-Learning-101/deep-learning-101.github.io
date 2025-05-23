---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---

<p align="center">
  <strong>Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101</strong>  
</p>
<p align="center">
  AI是一條孤獨且充滿惶恐及未知的旅程，花俏絢麗的收費課程或活動絕非通往成功的捷徑。<br>
  衷心感謝當時來自不同單位的AI同好參與者實名分享的寶貴經驗；如欲移除資訊還請告知。<br>
  由 <a href="https://www.twman.org/" target="_blank">TonTon Huang Ph.D.</a> 發起，及其當時任職公司(台灣雪豹科技)無償贊助場地及茶水點心。<br>
</p>  
<p align="center">
  <a href="https://huggingface.co/spaces/DeepLearning101/Deep-Learning-101-FAQ" target="_blank">
    <img src="https://github.com/Deep-Learning-101/.github/blob/main/images/DeepLearning101.JPG?raw=true" alt="Deep Learning 101" width="180"></a>
    <a href="https://www.buymeacoffee.com/DeepLearning101" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-red.png" alt="Buy Me A Coffee" style="height: 100px !important;width: 180px !important;" ></a>
</p>
<p align="center">
  <a href="https://www.youtube.com/@DeepLearning101" target="_blank">YouTube</a> |
  <a href="https://www.facebook.com/groups/525579498272187/" target="_blank">Facebook</a> |
  <a href="https://deep-learning-101.github.io/"> 回 GitHub Pages</a> |
  <a href="http://DeepLearning101.TWMAN.ORG" target="_blank">網站</a> |
  <a href="https://huggingface.co/DeepLearning101" target="_blank">Hugging Face Space</a>
</p>

# 第十四章 自編碼器

<a href="https://www.youtube.com/watch?v=5mrJmzzpPBs" target="_blank" rel="noopener noreferrer"><i class="fab fa-youtube mr-1"></i>2017/09/08, Autoencoders @ Deep Learning Book Chapter 14</a><br>


**重點摘要:**
自編碼器 (autoencoder) 是一種神經網路，經過訓練後嘗試將輸入複製到輸出。內部有一個隱藏層 `h`，可以產生編碼 (code) 來表示輸入。這個網路可以看作由兩部分組成：一個由函數 `h = f(x)` 表示的編碼器和一個生成重構的解碼器 `r = g(h)`。如果一個自編碼器只是簡單地學會將處設置為 `g(f(x)) = x`，那麼這個自編碼器就沒有什麼特別的用處。相反，自編碼器通常被設計成不能完美複製輸入，而是只能近似地複製，並且只對那些與訓練數據相似的輸入複製得很好。這種約束迫使其優先複製輸入的哪些部分，因此往往能學習到數據的有用特性。
現代自編碼器將編碼器和解碼器的概念推而廣之，將其中的確定函數推廣為隨機映射 `p_encoder(h|x)` 和 `p_decoder(x|h)`。
自編碼器的思想一直是神經網路歷史景象的一部分。傳統自編碼器被用於降維或特徵學習。近年來，自編碼器與潛變量模型理論的聯繫將自編碼器帶到了生成式建模的前沿。自編碼器可以被看作是前饋網路的一個特例，並且可以使用完全相同的技術進行訓練，通常使用小批量梯度下降法。不同於一般的前饋網路，自編碼器也可以使用**再循環 (recirculation)** 訓練，這種學習算法基於比較初始輸入的激活和重構輸入的激活。


**Q:** 什麼是自編碼器 (autoencoder)？它的基本結構和目標是什麼？

**A:** 自編碼器是一種神經網路，其目標是學習將輸入複製到輸出。它通常包含兩個主要部分：
1.  **編碼器 (Encoder):** 將輸入數據 `x` 映射到一個（通常是低維的）內部表示或編碼 `h`，即 `h = f(x)`。
2.  **解碼器 (Decoder):** 將內部表示 `h` 映射回重構的數據 `r`，試圖使其盡可能接近原始輸入 `x`，即 `r = g(h)`。
    整體目標是學習函數 `f` 和 `g`，使得 `g(f(x)) ≈ x`。

**Q:** 為什麼說一個「完美」複製輸入的自編碼器沒有太大用處？自編碼器通常被設計成具有什麼樣的特性？

**A:** 如果自編碼器能夠完美地複製任何輸入（即學到一個恆等函數），那麼它就沒有學習到關於數據的任何有用結構或潛在表示，只是簡單地將輸入傳遞到輸出。
    自編碼器通常被設計成**不能完美複製輸入**，而是只能**近似地複製**，並且這種近似複製的能力主要針對那些與**訓練數據相似的輸入**。通過施加某些約束（例如，限制隱藏層 `h` 的維度，或者對編碼/重構過程添加噪聲或正則化），迫使自編碼器學習輸入數據中最重要的特徵或變化模式，以便能夠有效地重構它。

**Q:** 自編碼器的概念是如何從確定性函數推廣到隨機映射的？

**A:** 傳統的自編碼器使用確定性的編碼函數 `h = f(x)` 和解碼函數 `r = g(h)`。現代自編碼器將這個概念推廣，允許編碼器和解碼器是隨機映射，即條件機率分佈：
*   **隨機編碼器:** `p_encoder(h|x)`，表示給定輸入 `x` 時，潛在編碼 `h` 的機率分佈。
*   **隨機解碼器:** `p_decoder(x|h)`，表示給定潛在編碼 `h` 時，重構數據 `x` 的機率分佈。
    這種推廣使得自編碼器能夠與更廣泛的生成模型（如變分自編碼器 VAE）聯繫起來。

**Q:** 什麼是「再循環 (recirculation)」訓練？它與標準的前饋網路訓練有何不同？

**A:** 「再循環」訓練是一種自編碼器的訓練方法，它涉及到比較網路對原始輸入的激活與對重構輸入的激活。具體來說，輸入 `x` 首先通過編碼器和解碼器得到重構 `r = g(f(x))`。然後，這個重構 `r` 可能會再次作為輸入送回網路，形成一個循環。學習算法會基於原始輸入和重構輸入（或它們在網路中的激活）之間的差異來調整參數。
    這與標準的前饋網路訓練（如反向傳播）不同，後者通常只進行一次前向傳播和一次反向傳播來計算損失和梯度，而不涉及將輸出重新送回輸入端。再循環訓練更強調模型對其自身輸出的內部一致性和穩定性。

---

## 14.1 欠完備自編碼器

**重點摘要:**
獲得有用特徵的一種方法是限制隱藏編碼 `h` 的維度比輸入 `x` 小，這種編碼維度小於輸入維度的自編碼器稱為**欠完備自編碼器 (undercomplete autoencoder)**。學習欠完備的表示將強制自編碼器捕捉訓練數據中最顯著的特徵。學習過程可以簡單地描述為最小化一個損失函數 `L(x, g(f(x)))` (公式 14.1)，其中 `L` 是一個損失函數，懲罰 `g(f(x))` 與 `x` 的差異，如均方誤差。
當解碼器是線性的且 `L` 是均方誤差時，欠完備自編碼器學會張開與主成分分析 (Principal Component Analysis, PCA) 相同的子空間。具有非線性編碼函數 `f` 和非線性解碼函數 `g` 的自編碼器能夠學習出比 PCA 更強大的非線性推廣。然而，如果自編碼器的容量（編碼器和解碼器的能力）過大，它仍然可能執行複製任務而捕捉不到關於數據分佈的任何有用信息，即使隱藏層維度很小。例如，它可以學習一個映射，將離散的訓練樣本索引 `i` 映射到 `x^(i)`，而沒有學習到數據的任何潛在結構。


**Q:** 什麼是欠完備自編碼器？它的主要特點是什麼？

**A:** 欠完備自編碼器是指其隱藏編碼層 `h` 的維度小於輸入層 `x` 的維度的自編碼器。主要特點是它通過信息瓶頸（即低維的隱藏層）來迫使模型學習數據的壓縮表示，從而捕捉數據中最重要的特徵。

**Q:** 訓練欠完備自編碼器的目標是什麼？損失函數通常是什麼形式？

**A:** 訓練欠完備自編碼器的目標是最小化重構誤差，即讓解碼器重構的輸出 `g(f(x))` 盡可能接近原始輸入 `x`。損失函數通常是衡量 `x` 和 `g(f(x))` 之間差異的函數，例如均方誤差 (Mean Squared Error, MSE)：
`L(x, g(f(x))) = ||x - g(f(x))||^2`

**Q:** 在什麼情況下，欠完備自編碼器的行為類似於主成分分析 (PCA)？它如何能學習到比 PCA 更強大的表示？

**A:** 當編碼器 `f` 和解碼器 `g` 都是線性的，並且損失函數 `L` 是均方誤差時，欠完備自編碼器學習到的隱藏表示所張成的子空間與 PCA 找到的主成分子空間是相同的。
    如果編碼器和/或解碼器是**非線性**的（例如，使用帶有非線性激活函數的多層神經網路），自編碼器就能夠學習到比線性 PCA 更強大、更複雜的非線性數據流形和表示。

**Q:** 即使是欠完備自編碼器，如果其編碼器和解碼器的容量過大，可能會出現什麼問題？

**A:** 如果編碼器和解碼器（儘管隱藏層維度受限）具有非常大的學習容量（例如，非常深或非常寬的非線性網路），它們仍然可能學會一種「作弊」的方式來完美地（或近乎完美地）重構訓練數據，而沒有真正學習到數據的任何有意義的潛在結構或泛化能力。例如，模型可能只是簡單地學習了一個從訓練樣本的索引到其自身的映射，而不是學習數據的壓縮表示。

---

## 14.2 正則自編碼器

**重點摘要:**
與其限制模型容量（如欠完備自編碼器中隱藏層的大小），不如使用**正則自編碼器 (regularized autoencoder)**，即使賦予它們較大的容量，也能鼓勵模型學習其他屬性（除了將輸入複製到輸出）。這些屬性包括表示的稀疏性、對輸入擾動的小導數、以及對噪聲的魯棒性。
正則自編碼器通常使用一個損失函數，包含重構項和一個正則化項：
`L(x, g(f(x))) + Ω(h)` (公式 14.2, 其中 `h = f(x)`)
或更一般地 `L(x, g(f(x))) + Ω(h, x)`

---

### 14.2.1 稀疏自編碼器

**重點摘要:**
稀疏自編碼器 (sparse autoencoder) 除了重構誤差外，還在損失函數中加入一個稀疏懲罰項 `Ω(h)`，通常作用於隱藏層的編碼 `h`。例如，可以使用 L1 正則化 `Ω(h) = λ Σ_i |h_i|` (公式 14.6) 來誘導稀疏性，或者使用 KL 散度來懲罰隱藏單元激活的平均值與一個小的目標稀疏度之間的差異。稀疏自編碼器通常用於學習特徵，以便用於其他任務（如分類）。一個只有單個隱藏層的稀疏自編碼器必須響應輸入的獨特統計特性，而不是簡單地充當恆等函數。通過這種方式學習到的表示可以包括與 PCA 或其他線性變換基函數捕獲到的元素不相似的元素。


**Q:** 什麼是正則自編碼器？它與欠完備自編碼器的主要區別是什麼？

**A:** 正則自編碼器是指那些除了最小化重構誤差之外，還在其損失函數中加入一個或多個正則化項的自編碼器。這些正則化項旨在鼓勵模型學習具有某些期望屬性（如稀疏性、對擾動的魯棒性等）的表示，即使模型的容量（例如隱藏層的大小）可能很大。
    與欠完備自編碼器的主要區別在於，欠完備自編碼器主要通過限制隱藏層的維度（即信息瓶頸）來迫使模型學習有用的表示，而正則自編碼器則通過在損失函數中添加明確的懲罰項來實現這一點，同時允許模型具有更大的容量。

**Q:** 稀疏自編碼器是如何工作的？它通過什麼方式誘導表示的稀疏性？

**A:** 稀疏自編碼器是一種正則自編碼器，它在訓練時除了最小化重構誤差外，還會懲罰隱藏層編碼 `h` 的非稀疏性。
    誘導稀疏性的方式通常有：
    1.  **L1 正則化:** 在損失函數中加入隱藏單元激活的 L1 范數 `λ Σ_i |h_i|`。L1 懲罰傾向於使許多隱藏單元的激活值變為零。
    2.  **KL 散度懲罰:** 假設我們希望每個隱藏單元的平均激活率（在整個訓練集上）達到一個預設的低水平（稀疏度參數 `ρ`）。可以計算每個隱藏單元的實際平均激活率 `ρ̂_j`，然後在損失函數中加入 `ρ` 和 `ρ̂_j` 之間的 KL 散度作為懲罰項。
    這迫使模型在表示輸入時只使用少數幾個活躍的隱藏單元。

**Q:** 相比於簡單的恆等函數，稀疏自編碼器學習到的表示有什麼潛在的優勢？

**A:** 由於稀疏性約束，稀疏自編碼器不能簡單地學習一個恆等函數來複製輸入。它被迫去發現輸入數據中更本質、更具有代表性的特徵，並用少數活躍的隱藏單元來編碼這些特徵。這可能導致：
*   **更具可解釋性的特徵:** 每個活躍的隱藏單元可能對應輸入數據的某個特定方面或模式。
*   **更好的泛化能力:** 通過提取更本質的特徵，模型可能更容易泛化到未見過的數據。
*   **學習到與線性方法（如PCA）不同的特徵:** 稀疏自編碼器（特別是帶有非線性激活的）能夠學習到數據的非線性特徵，這些特徵可能無法通過 PCA 等線性方法捕獲。

---

### 14.2.2 去噪自編碼器

**重點摘要:**
去噪自編碼器 (Denoising Autoencoder, DAE) 接收一個被部分損壞的輸入 `x̃`（通過某種隨機映射 `C(x̃|x)` 從原始數據 `x` 生成），並訓練模型從 `x̃` 中重構出原始的、未損壞的數據 `x`。其損失函數通常是 `L(x, g(f(x̃)))` (公式 14.9)。DAE 迫使模型學習輸入變數之間的依賴關係，以便能夠從損壞的版本中恢復它們。DAE 實際上是在學習數據流形的切線空間，或者說是在學習一個向量場，將損壞的點拉回到原始數據流形上。


**Q:** 什麼是去噪自編碼器 (DAE)？它的訓練目標與標準自編碼器有何不同？

**A:** 去噪自編碼器 (DAE) 是一種自編碼器，它不是直接從原始輸入 `x` 學習重構 `x`，而是從一個被隨機損壞（加噪）的輸入版本 `x̃` 中學習重構原始的、乾淨的輸入 `x`。
    其訓練目標是最小化原始數據 `x` 與從損壞輸入 `x̃` 重構得到的 `g(f(x̃))` 之間的差異，即 `L(x, g(f(x̃)))`。

**Q:** 去噪自編碼器是如何迫使模型學習數據的有用結構的？

**A:** 為了能夠從損壞的輸入 `x̃` 中成功地重構出原始數據 `x`，DAE 必須學習到：
1.  **輸入變數之間的依賴關係:** 如果某些變數被損壞，模型需要利用未損壞變數以及它們之間的統計關係來恢復被損壞的部分。
2.  **數據的潛在流形:** DAE 可以被看作是在學習數據點所在的低維流形。當輸入被噪聲推出流形時，DAE 學習一個映射，將其拉回到流形上。
    通過學習去除噪聲，模型被迫捕捉數據中更魯棒和本質的特徵。

**Q:** DAE 的學習過程與得分匹配 (score matching) 有什麼聯繫？

**A:** 理論上，當噪聲非常小時，訓練 DAE 來最小化均方重構誤差近似於最小化得分匹配的目標函數，即學習數據分佈的得分 `∇_x log p_data(x)`。得分匹配是一種無需估計配分函數即可學習能量模型的方法。因此，DAE 提供了一種間接學習數據分佈密度模型的方法。

---

### 14.2.3 懲罰導數作為正則項

**重點摘要:**
另一種正則化自編碼器的方法是懲罰編碼函數 `f(x)` 相對於輸入 `x` 的導數（的范數）。收縮自編碼器 (Contractive Autoencoder, CAE) 使用正則化項 `Ω(h,x) = λ Σ_i ||∇_x h_i||^2` (公式 14.11)，即懲罰隱藏單元激活關於輸入的 Jacobian 矩陣的 Frobenius 范數的平方。這鼓勵編碼函數在輸入空間的大部分區域對輸入的微小變化不敏感，即學習到的表示對輸入擾動具有局部不變性。這與去噪自編碼器有一定聯繫，因為兩者都試圖學習對輸入小擾動魯棒的表示。


**Q:** 收縮自編碼器 (CAE) 是如何實現正則化的？它的正則化項是什麼？

**A:** 收縮自編碼器 (CAE) 通過在其損失函數中加入一個懲罰項來實現正則化，該懲罰項懲罰編碼函數 `f(x)`（即隱藏層激活 `h`）相對於輸入 `x` 的導數的大小。具體來說，正則化項通常是編碼函數的 Jacobian 矩陣 `J_f(x) = ∇_x f(x)` 的 Frobenius 范數的平方，乘以一個正則化係數 `λ`：
`Ω(h,x) = λ ||J_f(x)||_F^2 = λ Σ_i Σ_j (∂h_i/∂x_j)^2` (類似公式 14.11)

**Q:** 懲罰編碼函數的導數有什麼直觀意義？它鼓勵模型學習什麼樣的表示？

**A:** 直觀意義是鼓勵編碼函數在輸入空間的大部分區域是「收縮的 (contractive)」或「局部平滑的」。這意味著如果輸入 `x` 發生微小的變化，其對應的隱藏表示 `h = f(x)` 的變化也應該很小。
    它鼓勵模型學習到對輸入的微小擾動不敏感的、具有局部不變性的表示。模型被迫只對那些能夠導致輸出（重構）發生顯著變化的輸入方向敏感。

**Q:** 收縮自編碼器與去噪自編碼器在目標上有何相似之處？

**A:** 兩者都旨在學習對輸入的微小擾動魯棒的表示。
*   **去噪自編碼器**通過顯式地向輸入添加噪聲，並訓練模型去除噪聲來實現這一點。
*   **收縮自編碼器**通過直接懲罰編碼函數對輸入的敏感度（即導數的大小）來實現這一點。
儘管方法不同，但最終目標都是學習能夠捕捉數據本質結構、同時對無關擾動不敏感的特徵。

---

## 14.3 表示能力、層的大小和深度

**重點摘要:**
自編碼器通常只有單層的編碼器和解碼器，但這不是必然的。實際上深度編碼器和解碼器能提供更多優勢。
*   **通用近似器:** 一個具有足夠容量的單隱藏層自編碼器（例如，隱藏單元數量可以任意大）理論上可以完美重構任何數據，但這並不意味著它能學習到有用的表示。
*   **深度的好處:**
    1.  **更高效的表示:** 深度自編碼器（即編碼器和/或解碼器是多層神經網路）可以用指數級更少的計算單元來表示某些複雜的函數，相較於淺層自編碼器。
    2.  **層次化特徵:** 深度結構允許模型學習數據的層次化表示，從低級別的簡單特徵到高級別的複雜特徵。
    3.  **更好的壓縮/降維:** 深度自編碼器可以更有效地將數據壓縮到一個非常低維的編碼空間，同時仍然能夠較好地重構數據。


**Q:** 為什麼說一個容量足夠大的淺層（單隱藏層）自編碼器理論上可以完美重構任何數據，但這並不一定意味著它學到了有用的表示？

**A:** 因為如果淺層自編碼器的隱藏層單元數量足夠多（例如，等於或大於輸入維度），並且編碼器和解碼器具有足夠的非線性能力，它可能只是簡單地學會了「記住」訓練數據，或者找到一種方式將每個訓練樣本映射到一個唯一的隱藏編碼，然後再從這個唯一的編碼完美地重構回原始樣本。這種情況下，模型並沒有學習到數據的任何潛在結構、規律或泛化能力，只是實現了一個複雜的恆等映射（或對訓練數據的查找表）。

**Q:** 相較於淺層自編碼器，深度自編碼器在表示能力和學習特性方面有哪些潛在優勢？

**A:**
1.  **更高效的表示複雜函數:** 對於某些類型的複雜數據變換或流形，深度自編碼器可能可以用比淺層自編碼器少得多的參數（或隱藏單元）來有效地表示它們。
2.  **學習層次化特徵:** 深度結構天然地適合學習數據的層次化表示。每一層可以在前一層特徵的基礎上學習更抽象、更複雜的特徵，這對於理解複雜數據（如圖像、語音、文本）非常重要。
3.  **更好的降維和壓縮:** 深度自編碼器通常能夠將數據更有效地壓縮到一個低維的潛在空間，同時保留更多的重要信息，從而實現更好的降維效果和重構質量。
4.  **可能更容易優化:** 雖然訓練非常深的網路有其挑戰，但在某些情況下，將複雜的映射分解為多個較簡單的層次化映射可能比直接學習一個非常複雜的淺層映射更容易優化。

---

## 14.4 隨機編碼器和解碼器

**重點摘要:**
自編碼器的本質是一個前饋網路，可以使用與傳統前饋網路相同的損失函數和輸出單元。如果輸入數據是實值的，通常使用均方誤差損失和線性輸出單元。如果輸入數據是二值的或位向量，通常使用 sigmoid 輸出單元和二元交叉熵損失。
本節將自編碼器與潛變量機率模型聯繫起來，引入**隨機編碼器**和**隨機解碼器**的概念。
*   **隨機解碼器:** 給定隱藏編碼 `h`，解碼器定義了一個條件分佈 `p_decoder(x|h)`。我們可以將自編碼器視為學習這個分佈。
*   **隨機編碼器:** 給定輸入 `x`，編碼器定義了一個條件分佈 `p_encoder(h|x)`。
整個模型定義了一個聯合分佈 `p_model(x,h) = p_encoder(h|x) p_decoder(x|h)` （如果這樣定義，則編碼器和解碼器必須對稱，這不常見）或者更常見的是 `p_model(x,h) = p_model(h) p_decoder(x|h)`，其中 `p_model(h)` 是潛變量的先驗分佈。
當我們考慮隨機編碼器和解碼器時，自編碼器的訓練目標可以與最大化 `log p_decoder(x|h)`（對於給定的 `h = f(x)`）或更一般的變分推斷目標（如 VAE 中的 ELBO）聯繫起來。


**Q:** 如何將確定性的自編碼器（`h=f(x)`, `r=g(h)`）與機率模型的概念聯繫起來？

**A:** 可以將確定性的自編碼器看作是隨機編碼器和隨機解碼器的一種特殊（退化）情況：
*   **確定性解碼器 `r=g(h)`** 可以被視為一個隨機解碼器 `p_decoder(x|h)`，其機率質量完全集中在 `g(h)` 點上（例如，一個均值為 `g(h)` 且方差趨於零的高斯分佈，或者一個參數為 `g(h)` 的伯努利分佈，如果 `x` 是二值的且 `g(h)` 是 sigmoid 輸出）。
*   **確定性編碼器 `h=f(x)`** 可以被視為一個隨機編碼器 `p_encoder(h|x)`，其機率質量完全集中在 `f(x)` 點上。

**Q:** 在引入隨機編碼器和解碼器的框架下，自編碼器的學習目標可以如何解釋？

**A:**
1.  如果我們有一個確定的編碼 `h = f(x)`，並且將解碼器視為定義了 `p_decoder(x|h)`，那麼自編碼器的學習目標（例如最小化重構誤差）可以被解釋為最大化條件對數概似 `log p_decoder(x|h=f(x))`。例如，如果 `p_decoder(x|h)` 是一個均值為 `g(h)` 的高斯分佈，那麼最大化其對數概似就等價於最小化均方誤差 `||x - g(h)||^2`。
2.  如果我們同時考慮隨機編碼器 `p_encoder(h|x)` 和隨機解碼器 `p_decoder(x|h)`，並且引入一個潛變量的先驗分佈 `p_model(h)`，那麼我們可以構建一個完整的生成模型 `p_model(x,h) = p_model(h) p_decoder(x|h)`。在這種情況下，`p_encoder(h|x)` 可以被視為對真實後驗 `p_model(h|x)` 的一個近似（變分推斷）。學習目標就變成了最大化證據下界 (ELBO)，如變分自編碼器 (VAE) 中所示。

**Q:** 為什麼在機率視角下，定義一個聯合分佈 `p_model(x,h) = p_encoder(h|x) p_decoder(x|h)` 通常不如 `p_model(x,h) = p_model(h) p_decoder(x|h)` 常見或自然？

**A:**
*   `p_model(x,h) = p_model(h) p_decoder(x|h)` 是一個標準的生成模型的定義方式：首先從先驗中抽取一個潛在表示 `h`，然後根據這個潛在表示生成可觀測數據 `x`。這符合我們對生成過程的直觀理解。
*   `p_model(x,h) = p_encoder(h|x) p_decoder(x|h)` 這種定義方式比較奇怪。如果將 `p_decoder(x|h)` 視為從 `h` 生成 `x` 的過程，那麼 `p_encoder(h|x)` 應該是後驗 `p(h|x)`。但如果我們將 `p_encoder(h|x)` 視為一個從 `x` 到 `h` 的因果過程，那麼 `p_decoder(x|h)` 就不再是標準的生成過程。更重要的是，如果這樣定義，為了使 `p_model(x,h)` 成為一個合法的聯合分佈（即對 `x` 和 `h` 積分/求和為1），通常需要 `p_encoder` 和 `p_decoder` 之間存在一種對稱性或特定的約束關係，這在實踐中很難滿足或不自然。
    在變分自編碼器的框架中，我們通常使用 `p_model(h)p_decoder(x|h)` 作為生成模型，而 `p_encoder(h|x)`（在 VAE 中記為 `q(h|x)`）被視為對真實後驗 `p_model(h|x)` 的一個近似。

---

## 14.5 去噪自編碼器

**重點摘要:**
本節從機率視角更深入地探討去噪自編碼器 (DAE)。DAE 訓練時的代價函數（公式 14.3）可以被看作是在最小化 `-E_{x~p̂_data(x)} E_{x̃~C(x̃|x)} log p_decoder(x|h=f(x̃))` (公式 14.14)。DAE 可以被視為在執行**得分估計 (score estimation)** 或 **得分匹配 (score matching)** 的一種形式。當噪聲很小時，DAE 的訓練目標近似於學習數據分佈的**得分 (score)**，即 `∇_x log p_data(x)`。

---

### 14.5.1 得分估計

**重點摘要:**
如果一個模型是確定性的，去噪自編碼器就是一個前饋網路，並且可以使用與其他前饋網路完全相同的方式進行訓練。得分匹配 (Hyvärinen, 2005a) 是最大概似的替代。它提供了概似梯度的另一種估計，即使模型在各個數據點 `x` 上獲得與數據分佈相同的得分（score）。在這種情況下，得分是一個特定的梯度場 `∇_x log p(x)` (公式 14.15)。DAE 的訓練準則（條件高斯 `p(x|h)`）能讓自編碼器學到估計數據分佈得分的向量場（`g(f(x̃)) - x̃`），這是 DAE 的一個重要特性。


**Q:** 在去噪自編碼器的上下文中，「得分 (score)」指的是什麼？DAE 如何與得分估計相關聯？

**A:** 在這裡，「得分」指的是數據的對數機率密度關於數據點 `x` 本身的梯度，即 `∇_x log p_data(x)`。
    去噪自編碼器 (DAE) 與得分估計的關聯在於，當向輸入數據 `x` 添加的噪聲非常小時，訓練 DAE 來從損壞的 `x̃` 中重構 `x`（通常是最小化均方誤差 `||x - g(f(x̃))||^2`）的目標函數，其梯度近似於一個與得分匹配相關的目標。更具體地說，DAE 學習到的重構方向 `g(f(x̃)) - x̃`（從損壞點指向重構點的向量）可以被看作是在估計真實數據分佈的得分向量 `∇_x log p_data(x)`。
    換句話說，DAE 試圖學習一個向量場，這個向量場指向數據流形密度增加最快的方向。

**Q:** 為什麼說DAE學習到的重構方向 `g(f(x̃)) - x̃` 可以用來估計數據分佈的得分？

**A:** 直觀地看，如果 `x̃` 是一個由於噪聲而偏離了高密度數據區域（數據流形）的點，那麼 DAE 的目標是將 `x̃` 拉回到高密度區域。這個「拉回」的向量 `g(f(x̃)) - x̃` 指向了密度增加的方向。數據分佈的得分 `∇_x log p_data(x)` 正是指向對數密度增加最快的方向的向量。因此，在噪聲較小的情況下，這兩個向量在方向上是相關的。更嚴格的數學推導（如 Vincent, 2011 中所示）可以證明這種聯繫。

---

## 14.6 使用自編碼器學習流形

**重點摘要:**
自編碼器與其他很多機器學習算法一樣，也利用了數據集中在一個低維流形（或少量流形的並集）上的思想。
*   **流形的切平面:** 流形的一個重要特性是它在每個點 `x` 處都有一個切平面（tangent plane）。DAE 的重構方向 `g(f(x̃)) - x̃` 提供了對這個切平面的近似。
*   **學習數據流形:** 自編碼器（特別是欠完備的或正則化的）試圖學習這個數據流形。它們的訓練過程和兩種推動力可以被這樣解釋：
    1.  學習訓練樣本 `x` 的表示 `h`，使得 `x` 能通過解碼器從 `h` 中重構。這意味著自編碼器不需要成功重構不屬於數據生成過程的點。
    2.  滿足約束或正則化項。這可以是限制 `h` 的維度，或者對編碼器/解碼器施加稀疏性、收縮性等。
*   **非參數流形學習:** 許多專門的流形學習算法（如 Isomap, LLE, Laplacian Eigenmaps）基於最近鄰圖。圖 14.8 展示了這種方法。
*   **自編碼器與流形:** 自編碼器通過學習一個（通常是參數化的）從輸入空間到低維潛在空間的映射（編碼），以及一個從潛在空間到輸入空間的映射（解碼），來間接學習數據流形。理想情況下，潛在空間捕獲了流形的內在維度和結構。


**Q:** 「數據流形 (data manifold)」假設在自編碼器的背景下指的是什麼？

**A:** 數據流形假設是指，儘管觀測到的高維數據（如圖像像素）可能位於一個非常高維的空間中，但這些數據點實際上並非均勻地分佈在整個空間，而是集中在一個或多個嵌入在該高維空間中的、具有較低內在維度的光滑子空間（即流形）上或其附近。
    在自編碼器的背景下，這意味著自編碼器試圖學習這個低維流形的結構，並找到一種方法將高維數據點有效地表示在這個流形上（通過編碼到潛在空間），或者從潛在空間生成位於這個流形上的數據點（通過解碼）。

**Q:** 去噪自編碼器 (DAE) 如何利用流形假設來進行學習？它學習到的重構與流形的切平面有何關係？

**A:** DAE 的工作原理與流形假設密切相關：
1.  當向一個位於數據流形上的乾淨數據點 `x` 添加噪聲得到 `x̃` 時，`x̃` 通常會偏離原始流形。
2.  DAE 的目標是從 `x̃` 中重構出 `x`，這相當於學習一個映射，將偏離流形的點「拉回」到流形上。
3.  這個「拉回」的方向 `g(f(x̃)) - x̃`（從損壞點指向重構點）可以被認為是近似於數據流形在 `x` 點（或其附近）的切平面（或切空間）的方向。因為切平面定義了流形在該點的局部線性近似，也是密度增加最快的方向（如果噪聲是各向同性的）。
    因此，DAE 通過學習去除噪聲，間接地學習了數據流形的局部幾何結構。

**Q:** 自編碼器與傳統的非參數流形學習算法（如 Isomap, LLE）在學習流形方面有何主要不同？

**A:**
*   **非參數流形學習算法 (如 Isomap, LLE):** 這些算法通常是「非參數的」，它們不學習一個明確的從輸入空間到低維嵌入空間的映射函數，也不學習一個從低維嵌入空間到原始數據空間的逆映射函數。它們主要基於數據點之間的局部鄰域關係（例如，構建最近鄰圖）來推斷流形的全局結構，並為訓練數據點找到一個低維嵌入。對於新的、未見過的數據點，通常需要額外的步驟（如基於插值）來計算其低維嵌入。
*   **自編碼器:** 自編碼器學習的是參數化的映射函數：編碼器 `f(x)`（從輸入到潛在空間）和解碼器 `g(h)`（從潛在空間到重構空間）。一旦訓練完成，對於任何新的輸入點，都可以通過編碼器直接計算其潛在表示。自編碼器試圖學習一個能夠描述整個數據流形的（參數化的）模型。
    簡而言之，非參數方法更側重於發現訓練數據本身的流形結構並進行嵌入，而自編碼器則試圖學習能夠處理新數據的、具有泛化能力的流形模型。

---

## 14.7 收縮自編碼器

**重點摘要:**
收縮自編碼器 (Contractive Autoencoder, CAE) (Rifai et al., 2011a,b) 在編碼 `h=f(x)` 的基礎上添加了顯式的正則項，鼓勵 `f` 的導數盡可能小：
`Ω(h) = λ ||∂f(x)/∂x||_F^2` (公式 14.18，Frobenius 范數)
這個正則項作用於編碼器的 Jacobian 矩陣。
*   **與去噪自編碼器的聯繫:** CAE 和 DAE 都試圖學習對輸入微小擾動不敏感的表示。當噪聲較小時，DAE 的重構準則 `r = g(f(x̃)) ≈ x` 近似於使 `f` 對 `x̃` 不敏感。
*   **收縮方向:** CAE 的目標是學習數據流形的切線方向。它鼓勵編碼器對沿著流形方向的變化敏感（以便重構），而對垂直於流形方向的變化不敏感（收縮這些方向）。
*   **局部 PCA:** 在某些情況下，CAE 的 Jacobian 矩陣的奇異向量可以解釋為局部 PCA 的基。
*   **實現:** 儘管 CAE 在理論上很有吸引力，但其 Jacobian 的計算成本可能較高。堆疊多個 CAE 形成深度網路也是一個研究方向。


**Q:** 收縮自編碼器 (CAE) 的正則化項是什麼？它試圖最小化什麼？

**A:** 收縮自編碼器 (CAE) 的正則化項是編碼函數 `f(x)`（即隱藏層激活 `h`）相對於輸入 `x` 的 Jacobian 矩陣的 Frobenius 范數的平方，乘以一個正則化係數 `λ`：
`Ω(h) = λ ||J_f(x)||_F^2 = λ ||∂f(x)/∂x||_F^2` (公式 14.18)
它試圖最小化隱藏表示對輸入變化的敏感度。

**Q:** CAE 學習到的表示具有什麼樣的特性？它與數據流形的幾何結構有何關係？

**A:** CAE 學習到的表示具有**局部不變性**，即當輸入 `x` 發生微小變化時，其對應的隱藏表示 `h=f(x)` 的變化也應該很小。
    與數據流形的幾何結構的關係在於，CAE 傾向於使編碼函數在**垂直於數據流形的方向上是收縮的**（即對這些方向的輸入變化不敏感），而在**沿著數據流形的方向上則相對不那麼收縮**（即對這些方向的輸入變化更敏感，以便能夠區分流形上的不同點並進行重構）。因此，CAE 學習到的特徵對那些不改變數據點在流形上本質位置的擾動是魯棒的。

**Q:** 計算 CAE 的正則化項（Jacobian 的 Frobenius 范數）在實踐中可能存在什麼問題？

**A:** 主要問題是計算成本。計算完整的 Jacobian 矩陣 `∂f(x)/∂x` 需要對每個隱藏單元關於每個輸入單元求偏導。如果輸入維度和隱藏層維度都很大，這個 Jacobian 矩陣會非常大，計算其 Frobenius 范數的平方會涉及到大量的偏導數計算和平方和運算，這在每次迭代中都可能非常耗時。對於非常深的編碼器，計算這個 Jacobian 也會更加複雜。

---

## 14.8 預測稀疏分解

**重點摘要:**
預測稀疏分解 (Predictive Sparse Decomposition, PSD) (Kavukcuoglu et al., 2008) 是一種混合模型，結合了稀疏編碼和自編碼器的思想。它被訓練為能夠預測輸入的稀疏編碼的輸出。PSD 被應用於圖片和視頻中對象識別的無監督特徵學習。這個模型由一個編碼器 `f(x)` 和一個解碼器 `g(h)` 組成，並且是參數化的。在訓練過程中，`h` 由算法控制。優化過程是最小化：
`||x - g(h)||^2 + λ||h||_1 + γ||h - f(x)||^2` (公式 14.19)
第一項是稀疏編碼的重構誤差，第二項是 L1 稀疏懲罰，第三項是預測誤差，鼓勵編碼器 `f(x)` 的輸出接近於通過優化前兩項得到的稀疏編碼 `h`。PSD 的訓練程序不是先訓練稀疏編碼模型，然後訓練 `f(x)` 來預測稀疏編碼的特征。而是同時學習所有三個方面。


**Q:** 什麼是預測稀疏分解 (PSD)？它結合了哪些模型的思想？

**A:** 預測稀疏分解 (PSD) 是一種學習特徵表示的模型，它結合了**稀疏編碼 (Sparse Coding)** 和**自編碼器 (Autoencoder)** 的思想。
    它既像稀疏編碼一樣，試圖找到輸入數據 `x` 的一個稀疏表示 `h`，使得 `x` 可以通過一個解碼器 `g(h)` 來重構；同時，它又像自編碼器一樣，學習一個編碼器（或預測器）`f(x)`，使其能夠直接從輸入 `x` 預測出這個稀疏表示 `h`。

**Q:** PSD 模型的損失函數包含哪些主要組成部分？它們各自的作用是什麼？

**A:** PSD 模型的損失函數通常包含三個主要組成部分 (公式 14.19)：
1.  **重構誤差項 `||x - g(h)||^2`:** 這是稀疏編碼部分的損失，鼓勵解碼器 `g(h)` 能夠從稀疏編碼 `h` 中準確地重構原始輸入 `x`。
2.  **稀疏懲罰項 `λ||h||_1`:** 這是稀疏編碼部分的 L1 正則化項，鼓勵潛在編碼 `h` 是稀疏的（即 `h` 的許多元素為零）。
3.  **預測誤差項 `γ||h - f(x)||^2`:** 這是自編碼器（或預測器）部分的損失，鼓勵編碼器 `f(x)` 的輸出盡可能接近於通過優化前兩項得到的「最優」稀疏編碼 `h`。
    參數 `λ` 和 `γ` 控制了稀疏性和預測準確性的相對重要性。

**Q:** PSD 的訓練過程與先訓練稀疏編碼再訓練預測器的方法有何不同？

**A:** PSD 的訓練過程是**同時學習**稀疏編碼的字典（通過解碼器 `g`）、輸入的稀疏表示 `h` 以及預測該稀疏表示的編碼器 `f`。它不是一個分階段的過程（例如，先固定字典和編碼，然後學習預測器），而是在一個統一的目標函數下聯合優化所有這些組件。在每次迭代中，通常會先固定 `f` 和 `g`，通過優化找到對應當前 `x` 的 `h`，然後再固定 `h`，更新 `f` 和 `g` 的參數。

---

## 14.9 自編碼器的應用

**重點摘要:**
自編碼器已成功應用於降維和信息檢索。
*   **降維與可視化:** Hinton and Salakhutdinov (2006) 展示了一個深度自編碼器（由堆疊的 RBM 初始化，然後用反向傳播微調）能夠學習到比 PCA 更好的數據（如 MNIST 手寫數字）的低維嵌入。例如，將 784 維的 MNIST 圖像壓縮到 30 維，然後再到 2 維，產生的 2 維表示比 PCA 產生的更能保持數據的類別結構。
*   **信息檢索 (Information Retrieval):** 低維表示可以提高許多任務的性能，例如小空間的模型的計算和運行時間。Salakhutdinov and Hinton (2007b) 和 Torralba et al. (2008) 觀察到，許多降維的形式會將語義上相近的樣本映射到潛在空間的鄰近位置。映射的嵌入有助於泛化。
    *   **語義哈希 (Semantic Hashing):** Salakhutdinov and Hinton (2007b, 2009b) 提出通過將自編碼器的二值編碼解釋為內存地址來實現快速信息檢索。如果語義相似的文檔具有相似的二值編碼（即 Hamming 距離近），則可以快速找到相似文檔。通常在最頂層使用 sigmoid 激活，並將輸出閾值化為 0 或 1。
自編碼器的思想也已在其他多個方向進一步探討，包括改變損失函數的懲罰或探索在哈希表示中查找附近樣本的更直接的聯繫。


**Q:** 自編碼器在降維方面有哪些成功的應用？它相比於傳統的線性降維方法（如PCA）有什麼優勢？

**A:**
*   **成功的應用:** Hinton and Salakhutdinov (2006) 的工作是一個經典例子，他們使用深度自編碼器對 MNIST 手寫數字圖像和文檔數據進行降維。結果表明，深度自編碼器學習到的低維表示（例如，將圖像降到 2 維或 3 維進行可視化）能夠比 PCA 更好地保留數據的非線性結構和類別信息。
*   **相比 PCA 的優勢:**
    1.  **非線性降維:** 自編碼器（特別是帶有非線性激活函數的）可以學習數據的非線性流形和複雜的非線性變換，而 PCA 只能進行線性投影。
    2.  **學習更具判別性的表示:** 由於其非線性能力，自編碼器學習到的低維表示可能比 PCA 的表示更具有判別性，即在低維空間中更容易區分不同的類別。
    3.  **層次化特徵提取:** 深度自編碼器可以學習數據的層次化特徵，這有助於捕捉不同抽象級別的數據結構。

**Q:** 什麼是語義哈希 (semantic hashing)？自編碼器如何用於實現語義哈希以進行快速信息檢索？

**A:** 語義哈希是一種將高維數據（如文檔、圖像）映射到緊湊的二值碼（哈希碼）的技術，其目標是使得語義上相似的數據點具有相似的（Hamming 距離近的）哈希碼。
    自編碼器可以用於實現語義哈希：
    1.  **訓練一個自編碼器:** 該自編碼器的隱藏編碼層（或一個專門的輸出層）被設計為輸出二值或接近二值的編碼。例如，可以使用 sigmoid 激活函數，然後對其輸出進行閾值化（如大於 0.5 則為 1，否則為 0）。
    2.  **生成哈希碼:** 對於每個數據點，通過訓練好的自編碼器的編碼器部分計算其二值編碼，這個編碼就作為該數據點的哈希碼。
    3.  **快速信息檢索:** 當需要查找與查詢數據點相似的項目時，首先計算查詢點的哈希碼，然後在數據庫中搜索具有相同或 Hamming 距離近的哈希碼的項目。由於哈希碼是二值的且維度較低，這種搜索可以非常快速（例如，通過哈希表）。
    關鍵在於自編碼器學習到的編碼能夠捕捉數據的語義信息，使得語義相似的項目被映射到相近的哈希碼。

---

希望這些詳細的摘要和Q&A對您有所幫助！