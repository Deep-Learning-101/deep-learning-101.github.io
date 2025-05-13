[https://www.twman.org/AI/NLP](https://www.twman.org/AI/NLP)

[https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper)

[https://huggingface.co/DeepLearning101](https://huggingface.co/DeepLearning101)

[https://deep-learning-101.github.io/](https://deep-learning-101.github.io/)

<details close>
<summary>手把手帶你一起踩 AI 坑</summary>

<h3><a href="https://blog.twman.org/p/deeplearning101.html">手把手帶你一起踩 AI 坑</a>：<a href="https://www.twman.org/AI">https://www.twman.org/AI</a></h3>

<ul>
  <li>
    <b><a href="https://blog.twman.org/2025/03/AIAgent.html">避開 AI Agent 開發陷阱：常見問題、挑戰與解決方案</a></b>：<a href="https://deep-learning-101.github.io/agent">探討多種 AI 代理人工具的應用經驗與挑戰，分享實用經驗與工具推薦。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/08/LLM.html">白話文手把手帶你科普 GenAI</a></b>：<a href="https://deep-learning-101.github.io/GenAI">淺顯介紹生成式人工智慧核心概念，強調硬體資源和數據的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/09/LLM.html">大型語言模型直接就打完收工？</a></b>：<a href="https://deep-learning-101.github.io/1010LLM">回顧 LLM 領域探索歷程，討論硬體升級對 AI 開發的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/07/RAG.html">檢索增強生成(RAG)不是萬靈丹之優化挑戰技巧</a></b>：<a href="https://deep-learning-101.github.io/RAG">探討 RAG 技術應用與挑戰，提供實用經驗分享和工具建議。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/02/LLM.html">大型語言模型 (LLM) 入門完整指南：原理、應用與未來</a></b>：<a href="https://deep-learning-101.github.io/0204LLM">探討多種 LLM 工具的應用與挑戰，強調硬體資源的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2023/04/GPT.html">什麼是大語言模型，它是什麼？想要嗎？(Large Language Model，LLM)</a></b>：<a href="https://deep-learning-101.github.io/GPU">探討 LLM 的發展與應用，強調硬體資源在開發中的關鍵作用。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/11/diffusion.html">Diffusion Model 完全解析：從原理、應用到實作 (AI 圖像生成)</a></b>；<a href="https://deep-learning-101.github.io/diffusion">深入探討影像生成與分割技術的應用，強調硬體資源的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2024/02/asr-tts.html">ASR/TTS 開發避坑指南：語音辨識與合成的常見挑戰與對策</a></b>：<a href="https://deep-learning-101.github.io/asr-tts">探討 ASR 和 TTS 技術應用中的問題，強調數據質量的重要性。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2021/04/NLP.html">那些 NLP 踩的坑</a></b>：<a href="https://deep-learning-101.github.io/nlp">分享 NLP 領域的實踐經驗，強調數據質量對模型效果的影響。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2021/04/ASR.html">那些語音處理踩的坑</a></b>：<a href="https://deep-learning-101.github.io/speech">分享語音處理領域的實務經驗，強調資料品質對模型效果的影響。</a>
  </li>
  <li>
    <b><a href="https://blog.twman.org/2020/05/DeepLearning.html">手把手學深度學習安裝環境</a></b>：<a href="https://deep-learning-101.github.io/101">詳細介紹在 Ubuntu 上安裝深度學習環境的步驟，分享實際操作經驗。</a>
  </li>
</ul>

</details>

# NLP
自然語言處理 (Natural Language Processing)

關於自然語言處理，如果你在臺灣，那你第一時間應該會想到俗稱Chatbot的聊天機器人 (就是要人工維運關鍵字跟正規表示式的機器人)吧？從最早的中英文的情感分析，陸續接觸過文本糾錯(校正)、文本分類、文本相似度、命名實體識別、文本摘要、機器閱讀理解等，當然自然語言處理其實博大精深，還有像是分詞、詞性標註、句法分析、語言生成等，傳說中的知識圖譜 (Ontology？) 更是大魔王了；這邊僅先匯整接觸過的做說明，當然在深度學習還未爆紅前，已經有非常多的演算法，未來也盡量針對各個項目與領域持續更新簡單介紹，就當近幾次專題演講的摘要，也算是這幾年跟小夥伴們奮鬥NLP充滿血與淚的回憶；另外，根據經驗，論文當然要追，更要實作跟實驗，但算法模型其實效果已經都差不多，如果你想將算法實際落地，別懷疑，請好好的處理你的數據，這會是蠻關鍵的地方。另外，你一定也要知道 BERT家族，早在2018年11月，Google 大神釋出 BERT 後，就差不多屌打各種自然語言處理應用 (在這之前，你想搞自然語言處理，勢必用到騰訊所開源需要16GB記憶體的Tencent_ChineseEmbedding)，再後來還有像 transformer 跟 huggingface，所以你一定要花點時間瞭解；當然，如果你真的沒太多時間投入去換跟處理數據然後重新訓練，那歡迎聯絡一下，用我們還持續迭代開發的臺灣深度大師啦，不然公開數據都是對岸用語或簡體跟英文還要擠GPU計算資源，你會很頭痛 ! 對啦，你也可以試試 NVIDIA GTC 2021 介紹的Javis等對話式AI等東西，但我想你還是會覺得不容易上手就是，除非你想自己從頭硬幹去瘋狂的標註適合自己場景的數據，然後瞭解怎樣重新訓練模型。

<b>2018/07/15-2020/02/29 開發心得</b>

自然語言處理（英語：Natural Language Processing，縮寫作 NLP）是人工智慧和語言學領域的分支學科。此領域探討如何處理及運用自然語言；自然語言處理包括多方面和步驟，基本有認知、理解、生成等部分。 自然語言認知和理解是讓電腦把輸入的語言變成有意思的符號和關係，然後根據目的再處理。自然語言生成系統則是把計算機數據轉化為自然語言。最後，放眼望去想入門 Attention、Transformer、Bert 和 李宏毅老師的教學影片等，絕對不能錯過。 雖然分享這些踩過的坑還有免費DEMO跟API其實我想不到有啥好處，但至少不用為了要營利而去亂喊口號也更不用畫大餅，能做多少就是說多少；如同搞 Deep Learning 101 搞那麼久，搬桌椅、直播場佈其實比想像中麻煩，只希望讓想投入的知道 AI 這個坑其實很深，多分享總是比較好 !

<ul>
<li>
  <b><a href="https://mp.weixin.qq.com/s/SJXxeTsqn9RoaVu66MISXQ">這麼多年，終於有人講清楚Transformer了</a></b>
</li>
<li>
  <b><a href="https://zhuanlan.zhihu.com/p/411311520">我從零實現了Transformer模型，把代碼講給你聽</a></b>
</li>
<li>
  <b><a href="https://easyai.tech/ai-definition/attention/">Attention 機制</a></b>
</li>
<li>
  <b><a href="https://zhuanlan.zhihu.com/p/410776234">超詳細圖解Self-Attention</a></b>
</li>
<li>
  <b><a href="https://huggingface.co/learn/nlp-course/zh-TW/chapter1/1">NLP Course @ HuggingFace</a></b>
</li>
</ul>

<details close>
  <summary>Information/Event Extraction (資訊/事件擷取)</summary>
<ul>
  <li>
    <b><a href="https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper/blob/main/HugNLP.md">HugNLP</a></b>：
    <a href="https://blog.twman.org/2023/07/HugIE.html">以 MRC 為核心的統一信息抽取框架，支援醫療應用如診斷書與醫囑擷取。</a>
  </li>
  <li>
    <b><a href="https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper/blob/main/DeepKE.md">DeepKE</a></b>：
    <a href="https://github.com/zjunlp/DeepKE/blob/main/README_CN.md">支援中文知識圖譜抽取，包含 DeepKE-LLM 與 KnowLM 擴展模組。</a>
  </li>
  <li>
    <b><a href="https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper/blob/main/ERNIE-Layout.md">ERNIE-Layout</a></b>：
    <a href="https://arxiv.org/abs/2210.06155">增強視覺結構理解的預訓練模型，提升文件排版感知能力。</a>
  </li>
  <li>
    <b><a href="https://huggingface.co/spaces/DeepLearning101/PaddleNLP-UIE">UIE @ PaddleNLP</a></b>：
    <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie">支援任意類型信息抽取任務的開源工具。</a>
  </li>
</ul>
</details>

<details close> 
  <summary>Machine Reading Comprehension (機器閱讀理解)</summary>
  
    <b>2018/10/15–2019/02/10 開發心得：</b><br>
    
    投入約 120 天，開發用於博物館與展場導覽機器人的問答系統。當時缺乏中文資料集，無法直接套用英文 SQuAD 1.0/2.0。初期需自行翻譯資料、自建標註系統，並標註維基百科語料以彌補在地語言差異。挑戰包含多篇文章處理、跨文件推理，以及中英文格式差異與語境適配。
   
<ul>
  <li><a href="https://www.twman.org/AI/NLP/MRC">中文機器閱讀理解</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/80905984">機器閱讀理解綜述(一)</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/80980403">機器閱讀理解綜述(二)</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/81126870">機器閱讀理解綜述(三)</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/109309164">機器閱讀理解探索與實踐</a></li>
</ul>  
</details>    

<details close>
  <summary>Named Entity Recognition (命名實體識別)</summary>
  
  <b>2019/12/02-2020/02/29 開發心得</b><br>

記得前後兩次陸續投入總計約100天。或許有人會發現為何在分享這幾篇自然語言會強調中文數據？最好理解的說法就是中文是基於字表示再加上中文斷詞的效果，比起每個單詞只需空格來表示的英文硬是麻煩點。命名實體識別 (Named Entity Recognition, NER) 是指將語句中的元素分成預先定義的類別 (開放域來說包括實體、時間和數字3個大類，人名、地名、組織名、機構名、時間、日期、數量和名字等7個小類，特定領域就像是藥名、疾病等類別)。要應用在那方面？像是關係抽取、對話意圖理解、輿情分析、對話NLU任務等等都用得上，更廣義的就屬填槽 (Slot-Filling) 了。
最早 (2019/08時) 我們需處理的場景是針對電話助理的對話內容 (就是APP幫你接電話跟對方對話) 在語音識別後跟語音合成前的處理，印像中沒做到非常深入；後來剛好招聘到熟悉NER這部份的小夥伴們，剛好一直想把聊天對話做個流程處理 (多輪對話的概念) ，就再花了點時間當做上手，因為不想依賴大量關鍵字和正規表示式做前處理，中間試了不少數據集，還做了像是用拼音、注音等，或者品牌定義等超多的實驗，甚至還一度想硬整合 RASA 等等的開源套件，也嘗試用了 "改寫" 來修正對話內容，去識別出語句中的重點字。至於這個的數據標據就真的更累人，意外找到一個蠻好用的標註系統 ChineseAnnotator，然後我們就瘋狂開始標註 !

<ul>
  <li><a href="https://www.twman.org/AI/NLP/NER">中文命名實體識別</a></li>
</ul> 
</details>

<details close>
  <summary>Correction (糾錯)</summary>
  
  <b>2019/11/20-2020/02/29 開發心得</b><br>

投入約100天，早期上線成本資源頗高，現在就沒這問題；這個項目堪稱是在NLP這個坑裡投入第二多的，記得當時的場景是機器人在商場裡回答問題所顯示出來的文字會有一些ASR的錯字，但是問題一樣卡在數據集，還因此讓小夥伴們花了好長時間辛苦去標註 XD，但看看現在效果，我想這是值得的 ! 記得一開始是先依賴 pycorrector，然後再換 ConvSeq2Seq，當然 bert 也做了相關優化實驗，中間一度被那三番二次很愛嗆我多讀書，從RD轉職覺得自己很懂做產品的PM拿跟百度對幹，從一開始的看實驗結果輸，到後來贏了，卻又自己亂測說還是不夠好之類的叭啦叭啦，說實話，你最後不也人設垮了然後閃人 ~ 攤手 ~ 
現在看看這截圖效果，不是蠻勵害的嗎 ? 真的想說這社會真的充滿一堆人設嚇死人的人，無敵愛嘴砲 ! 搞的為了滿足那位人設比天高的需求，真的是想了像是用拼音還兼NER來整合的好幾種方法 ! 那文本糾錯會有什麼坑呢？：數據啊、格式啊 !!! 還有幾個套件所要處理的目標不太一樣，有的可以處理疊字有的可以處理連錯三個字，還有最麻煩的就是斷字了，因為現有公開大家最愛用的仍舊是Jieba，即便它是有繁中版，當然也能試試 pkuseg，但就是差了點感覺。

<ul>
  <li><a href="https://www.twman.org/AI/NLP/Correction">中文文本糾錯</a></li>
  <li><a href="https://huggingface.co/spaces/DeepLearning101/Corrector101zhTW">HuggingFace Space</a></li>  
</ul> 
</details>

<details close>
  <summary>Classification (分類)</summary>

 <b>2019/11/10-2019/12/10 開發心得 </b><br>
 
最早我們是透過 Hierarchical Attention Networks for Document Classification (HAN) 的實作，來修正並且以自有數據進行訓練；但是這都需要使用到騰訊放出來的近16 GB 的 embedding：Tencent_AILab_ChineseEmbedding_20190926.txt，如果做推論，這會是個非常龐大需載入的檔案，直到後來 Huggingface 橫空出世，解決了 bert 剛出來時，很難將其當做推論時做 embedding 的 service (最早出現的是 bert-as-service)；同時再接上 BiLSTM 跟 Attention。CPU (Macbook pro)：平均速度：約 0.1 sec/sample，總記憶體消耗：約 954 MB (以 BiLSTM + Attention 為使用模型)。
引用 Huggingface transformers 套件 bert-base-chinese 模型作為模型 word2vec (embedding) 取代騰訊 pre-trained embedding
優點：API 上線時無須保留龐大的 Embedding 辭典,避免消耗大量記憶體空間，但BERT 相較於傳統辭典法能更有效處理同詞異義情況，更簡單且明確的使用 BERT 或其他 Transformers-based 模型
缺點：Embedding後的結果不可控制，BERT Embedding 維度較大,在某些情況下可能造成麻煩

<ul>
  <li><a href="https://www.twman.org/AI/NLP/Classification">中文文本分類</a></li>
</ul> 
  </details>


<details close>
  <summary>Similarity (相似度)</summary>
  
 <b>2019/10/15-2019/11/30 開發心得</b><br>

投入約45天，那時剛好遇到 albert，但最後還是被蒸溜給幹掉；會做文本相似度主要是要解決當機器人收到ASR識別後的問句，在進到關鍵字或正規表示式甚至閱讀理解前，藉由80/20從已存在的Q&A比對，然後直接解答；簡單來說就是直接比對兩個文句是否雷同，這需要準備一些經典/常見的問題以及其對應的答案，如果有問題和經典/常見問題很相似，需要可以回答其經典/常見問題的答案；畢竟中文博大精深，想要認真探討其實非常難，像是廁所在那裡跟洗手間在那，兩句話的意思真的一樣，但字卻完全不同；至於像是我不喜歡你跟你是個好人，這就是另一種相似度了 ~ xDDD ! 那關於訓練數據資料，需要將相類似的做為集合，這部份就需要依賴文本分類；你可能也聽過 TF-IDF 或者 n-gram 等，這邊就不多加解釋，建議也多查查，現在 github 上可以找到非常的範例程式碼，建議一定要先自己動手試試看 !

<ul>
  <li><a href="https://www.twman.org/AI/NLP/Similarity">中文文本相似度</a></li>
</ul> 
</details>
