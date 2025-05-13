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
    <b>2018/10/15–2019/02/10 開發心得：</b>
    投入約 120 天，開發用於博物館與展場導覽機器人的問答系統。當時缺乏中文資料集，無法直接套用英文 SQuAD 1.0/2.0。初期需自行翻譯資料、自建標註系統，並標註維基百科語料以彌補在地語言差異。挑戰包含多篇文章處理、跨文件推理，以及中英文格式差異與語境適配。
   
<ul>
  <li><a href="https://www.twman.org/AI/NLP/MRC">中文機器閱讀理解</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/80905984">機器閱讀理解綜述(一)</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/80980403">機器閱讀理解綜述(二)</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/81126870">機器閱讀理解綜述(三)</a></li>
  <li><a href="https://zhuanlan.zhihu.com/p/109309164">機器閱讀理解探索與實踐</a></li>
  <li><a href="https://communeit.medium.com/%E4%BB%80%E9%BA%BC%E6%98%AF%E6%A9%9F%E5%99%A8%E9%96%B1%E8%AE%80%E7%90%86%E8%A7%A3-%E8%B7%9F%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E6%9C%89%E4%BB%80%E9%BA%BC%E9%97%9C%E4%BF%82-b02fb6ccb6e9">什麼是機器閱讀理解？</a></li>
</ul>  
</details>    
