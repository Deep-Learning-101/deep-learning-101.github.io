---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
description: 大語言模型 (Large Language Model)。
permalink: /Large-Language-Model
lang: zh-Hant
schema_type: service
service_type: GenAI Consulting
---

{% include header.html %}

---

{% include ai-share.html %}

---

{% include price.html %}

---

# LLM  
大語言模型 (Large Language Model)


- [**AlpacaEval Leaderboard**](https://tatsu-lab.github.io/alpaca_eval/)
- [**Open LLM Leaderboard**](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [**Big Code Models Leaderboard**](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)

---

- [**Awesome-Chinese-LLM**](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)

---

- [大模型所需GPU 記憶體筆記](https://mp.weixin.qq.com/s/M_hdtR7mVq14MnaaL0MAUw)
- [不同參數規模大語言模型在不同微調方法下所需的顯存總結](https://www.datalearner.com/blog/1051703254378255)  

---

- [人工智慧大語言模型微調技術：SFT 監督微調、LoRA 微調方法、P-tuning v2 微調方法、Freeze 監督微調方法](https://zhuanlan.zhihu.com/p/643941480)
- [LoRA、完全微調到底有何不同？ MIT 21頁論文講懂了](https://www.jiqizhixin.com/articles/2024-11-11-5)  
- [大模型微調（Fine-tuning）全解，需要了解的都在這裡](https://www.53ai.com/news/finetuning/2025022604125.html)
- [初學者必看大模型微調指南：Unsloth官方微調技巧大公開！](https://mp.weixin.qq.com/s/COZfH_h36nX33TZGBVn0rg)
- [零代碼！一站式完整資料集準備到模型微調全流程！](https://zhuanlan.zhihu.com/p/1906670241645322809)
- [把你的De​​ePseek-R1 微調為某個領域的專家？](https://zhuanlan.zhihu.com/p/25054526736)
- [使用 NVIDIA NeMo 框架進行 LLM 模型剪枝和知識蒸餾](https://developer.nvidia.com/zh-cn/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)：https://huggingface.co/spaces/hiyouga/LLaMA-Board
   - [官方 README_zh.md](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)
   - 2024-09-13：[解析Llama-Factory：從微調到推理的架構](https://mp.weixin.qq.com/s/eJqKc_2nHBYzDFAp2AYdWQ)
   - [單卡3 小時訓練專屬大模型Agent：基於LLaMA Factory 實戰](https://zhuanlan.zhihu.com/p/678989191)  
- torchtune：[https://github.com/pytorch/torchtune](https://github.com/pytorch/torchtune)
   - [https://pytorch.dev.org.tw/torchtune/stable/index.html](https://pytorch.dev.org.tw/torchtune/stable/index.html)
   - [使用知識蒸餾將Llama3.1 8B 蒸餾到Llama3.2 1B](https://pytorch.ac.cn/torchtune/stable/tutorials/llama_kd_tutorial.html)  

---

- [微調特定領域的大模型，資料集究竟要怎麼搞？](https://zhuanlan.zhihu.com/p/29522986573)
- [LLaMA Factory 微調教學：如何建立高品質資料集](https://zhuanlan.zhihu.com/p/1916489160333714285)
- [大模型微調資料集生產工具 Easy Dataset](https://zhuanlan.zhihu.com/p/1908313086064042177)
- [開源DeepWiki版支援根據現有檔案產生微調資料集](https://zhuanlan.zhihu.com/p/1908831694879985815)
- [零一萬物發布COIG-CQIA：高品質且符合人類互動行為的中文指令微調數據](https://zhuanlan.zhihu.com/p/694434197)

---

- [AI Search Has A Citation Problem](https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php)
- [AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges](https://www.alphaxiv.org/abs/2505.10468)
- [OWASP Agentic AI – Threats and Mitigations](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/)

---

|  Tool 名稱 | 功能範疇 | 集成能力 | 適用場景 | 建議選擇 | 知識庫技術 |
|------------------|----------|----------|----------|----------|------------|
| **[Flowise](https://github.com/FlowiseAI/Flowise)** | 簡單可視化流程建構 | 支持 LangChain 和 GPT，多平台部署 | 快速構建簡單 LLM 流程應用 | 適合需要快速構建和部署 LLM 應用的用戶 | |
| **[Langflow](https://github.com/logspace-ai/langflow)** | 多智能體與 RAG 應用構建 | 外部工具、API 與資料庫集成 | 複雜任務鏈與多智能體應用開發 | 適合構建複雜任務鏈的技術型開發者 | |
| **[Dify](https://github.com/langgenius/dify)** | 全面應用開發與管理 | 多模型支持，全面工作流與模型管理 | 全生命周期管理與複雜應用開發 | 適合需要全面管理 LLM 應用的開發者或企業 | 基於 Pinecone 的向量數據庫、Notion API 同步（增量更新）、支持 Rerank 模型（bge-reranker-base）、提供行業模板庫 |
| **[n8n](https://github.com/n8n-io/n8n)** | 通用自動化與流程編排平台 | 支持超過 350 種服務與 API 集成，可視化流程編輯 | 應用整合、自動化工作流程構建、自動回應觸發器 | 適合需要自動處理非 AI 任務或整合各類 SaaS 工具的用戶與開發者 | 可與向量資料庫結合使用，但非內建 |
| **[RAGFlow](https://github.com/Dataland-Cloud/ragflow)** | 模組化 RAG 管線與知識應用框架 | 支持 LangChain、Chroma、FAISS 等，可用於構建完整 RAG 工作流 | 知識問答、文件檢索、RAG 多階段優化 | 適合需要建構可定製、模組化 RAG 系統的開發者 | 支持多向量資料庫（Chroma、FAISS）、可結合自定義資料源與檢索策略 |
| **[New API](https://github.com/Calcium-Ion/new-api)** | 模型接口統一與分發 | OpenAI 格式統一，支持多支付協議與分發管理 | 多模型接口管理與分發 | 適合需要統一管理多種 AI 模型接口的用戶 | |
| **[XORBITS Inference](https://github.com/xorbitsai/inference)** | 分散式推理與部署 | 與 Hugging Face 等模型相容，支援雲端及本地等多種部署環境 | 大規模模型推理與雲端部署，需快速搭建可擴展的推理服務時 | 適合需要高效擴展能力、進行大規模模型推理的團隊或企業 | |
| **[Ollama](https://github.com/jmorganca/ollama)** | 本地模型推理與管理 | 提供命令列介面，支援多種 Llama 模型於本地運行 | 在有隱私或離線需求的場景下進行本地推理 | 適合想在本地快速配置 Llama 系列模型的個人或中小型團隊 | |
| **[FastGPT](https://github.com/labring/FastGPT)** | 知識庫問答系統與工作流編排 | 提供開箱即用的數據處理、模型調用，支持可視化工作流編排 | 快速構建智能問答系統與複雜問答場景 | 適合需要快速部署知識庫問答系統的用戶 | 混合索引（Elasticsearch + FAISS）、支援 PDF 表格解析（PyMuPDF 集成）、BM25 權重可調、需人工標註種子數據 |
| **[Coze](https://github.com/cozeshow/coze)** | AI 智能體開發平台 | 整合插件、長短期記憶、工作流、卡片等功能，支持多平台發布 | 低門檻快速搭建個性化或具備商業價值的智能體 | 適合無需編程即可創建智能體的用戶 | 自研分佈式索引、電商數據結構 |

**工具**
---

- MCP
   - 2025-08-20：[FastAPI-MCP](https://zread.ai/tadata-org/fastapi_mcp)；[幾行程式碼即可將FastAPI 介面升級為MCP 工具服務](https://mp.weixin.qq.com/s/L568EP2tl2zwmC8vxz8s7w)
   - 2025-04-15：[automcp](https://github.com/NapthaAI/automcp)：[將任何代理、工具或編排器秒設定 MCP 伺服器](https://mp.weixin.qq.com/s/x-aZEhtnRYPFno81Fb9ttw)
   - 2025-04-10：[line-bot-mcp-server](https://github.com/line/line-bot-mcp-server)
   - 2025-04-05：[GitMCP](https://github.com/idosal/git-mcp)：[GitMCP 太神了！一行URL 讓AI 秒懂你的GitHub 項目](https://www.53ai.com/news/RAG/2025040590146.html)
   - 2025-03-14：[playwright-mcp](https://github.com/microsoft/playwright-mcp)：[MCP-Playwright：AI自動化神器](https://zhuanlan.zhihu.com/p/30178146112)   
- [Browser-use](https://github.com/browser-use/browser-use)
   - 2025-06-04：[workflow-use](https://github.com/browser-use/workflow-use)：[基於AI 的瀏覽器自動化工具：一次錄製，永久重複使用](https://zhuanlan.zhihu.com/p/1908094875066413718)
   - 2025-04-16：[browser-use/web-ui](https://github.com/browser-use/web-ui)；[如何使用](https://deepwiki.com/search/_bfd33aa8-cd79-4f1d-a1e8-5620d4374329)
   - 2025-03-28：[browser-use-webui](https://github.com/browser-use/web-ui)
   - 2025-02-16：[browser use webui部署（實現瀏覽器自動化）](https://zhuanlan.zhihu.com/p/24116360552)
   - 2025-01-23：[Browser Use – 讓AI 像人類一樣使用瀏覽器](https://zhuanlan.zhihu.com/p/20038156945)

---

- 2025-11-20：[LinearRAG](GitHub：https://github.com/DEEP-PolyU/LinearRAG)；[全新RAG框架LinearRAG:無需關係抽取，高效又精準！超越GraphRAG和LightRAG](https://zhuanlan.zhihu.com/p/1975321777342260763)
- 2025-09-11：[DeepMCPAgent](https://zread.ai/cryxnet/DeepMCPAgent)；[DeepMCPAgent 教你如何讓模型自己「找工具」！](https://mp.weixin.qq.com/s/Sj_7i1mTJ9WYaTlCzIqCFA)
- 2025-07-30：[langextract](https://github.com/google/langextract)；[隆重推出 LangExtract：由 Gemini 驅動的資訊擷取庫](https://developers.googleblog.com/zh-hans/introducing-langextract-a-gemini-powered-information-extraction-library/)
- 2025-07-26：[presenton](https://github.com/presenton/presenton)；[一款可本地部署的開源AI PPT項目，一鍵生成精美PPT](https://mp.weixin.qq.com/s/QTMVGD_aP41qrwtbjLxV8Q)
- 2025-07-03：[MultiAgentPPT](https://github.com/johnson7788/MultiAgentPPT)；[A2A+ADK+MCP多智能體並發系統產生(可線上編輯）的PPT（含原始碼）](https://zhuanlan.zhihu.com/p/1920611446007497267)
- 2025-06-28：[docext](https://github.com/NanoNets/docext)：[基於Qwen2.5VL的文檔解析工具](https://zhuanlan.zhihu.com/p/1919760450024879687)
- 2025-06-10：[Agentic-Doc](https://github.com/landing-ai/agentic-doc)；[LandingAI開源神器，這個Python庫讓百頁文檔秒變結構化資料！](https://zhuanlan.zhihu.com/p/1914259475306612709)
- 2025-06-06：[daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced)：每日自動爬取arXiv論文並以LLM產生中文摘要
- 2025-05-22：[AingDesk](https://deepwiki.com/aingdesk/AingDesk)；[AingDesk：零门槛本地 AI 部署](https://zhuanlan.zhihu.com/p/29773848356)
- 2025-05-20：[news-agents](https://deepwiki.com/eugeneyan/news-agents)
- 2025-05-16：[Follow](https://deepwiki.com/RSSNext/Folo)；[連續登頂GitHub 的資訊聚合神器：Follow，讓你不再錯過任何重要資訊！](https://zhuanlan.zhihu.com/p/1906505020628795653)
- 2025-05-11：[SurfSense](https://github.com/MODSetter/SurfSense)：[GitHub 開源專案 打通Notion、GitHub、搜尋引擎的AI超腦](https://mp.weixin.qq.com/s/kMhidgb6GkKEsl-D-u_7iw)，[如何使用](https://deepwiki.com/search/_df4a192b-a253-4155-a2a2-4a6fda9037e9)
- 2025-04-28：[PaperCoder](https://deepwiki.com/going-doer/Paper2Code)；[Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](https://www.alphaxiv.org/overview/2504.17192)
- 2025-04-16：[OneFileLLM](https://github.com/jimmc414/onefilellm)：[這個開源神器終結了AI資料整合的惡夢！一鍵聚合網頁、程式碼、論文到剪貼簿！](https://mp.weixin.qq.com/s/qNYX65fw-IWzEBLZpuaY6Q)
- 2025-04-16：[ScrapeGraphAI](https://github.com/ScrapeGraphAI/Scrapegraph-ai)：[ScrapeGraphAI －自然語言驅動的智慧爬蟲革命！](https://mp.weixin.qq.com/s/lQukAy12V5K1cH6rTkqxaA)
- 2025-04-15：[stagehand](https://github.com/browserbase/stagehand)：[Stagehand：AI驅動的下一代瀏覽器自動化框架](https://mp.weixin.qq.com/s/KF-z67kn4rTjcIBmTvj3nA)
- 2025-04-11：[nanobrowser](https://github.com/nanobrowser/nanobrowser)：[AI 驅動的瀏覽器自動化神器](https://mp.weixin.qq.com/s/65SwCtDta1cKvx1_BbaoHQ)
- 2025-04-10：[DevDocs](https://github.com/cyberagiinc/DevDocs)：[開發者的文檔收割機來了！這個開源工具讓你一小時幹完一週的活！](https://mp.weixin.qq.com/s/k5fG_L1q_19ylKIJD6PXmw)
- 2025-04-06：[sqlchat](https://github.com/sqlchat/sqlchat)：[這款開源神器讓資料庫管理像聊天一樣簡單！](https://mp.weixin.qq.com/s/kieSzWn3QDYvZ5Zx35hr1A)
- 2025-03-26：[pdf-craft](https://github.com/oomol-lab/pdf-craft)：[PDF秒轉Markdown/EPUB](https://zhuanlan.zhihu.com/p/1888288260171744707)
- 2025-03-25：[OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)；[OCRmyPDF 能力分析](https://www.zhihu.com/tardis/zm/art/32745781279?source_id=1003)
- 2025-03-12：[AingDesk](https://deepwiki.com/aingdesk/AingDesk)；[零門檻本地AI 部署，DeepSeek、Llama 一鍵直達！](https://zhuanlan.zhihu.com/p/29773848356)
- 2025-03-08：[composio](https://github.com/ComposioHQ/composio)：[AI助理效率神器！ Composio幫你輕鬆整合200+工具](https://mp.weixin.qq.com/s/rRPOmihGzcIXx0HQc3pdoA)
- 2025-02-25：[PySpur](https://www.pyspur.dev/)：[拖曳式開發AI工作流程！這款開源神器，讓LangChain也黯然失色！](https://zhuanlan.zhihu.com/p/26161709083)
- 2025-01-13：[DocAligner](https://github.com/ZZZHANG-jx/DocAligner)：[拍照文件復原：校正扭曲、光照陰暗、版面定位](https://mp.weixin.qq.com/s/Bra9h3ExddB5NiH1g4uk1g)
- 2025-01-13：[PPTAgent](https://github.com/icip-cas/PPTAgent)；[中科院開源AI工具，自動將文件轉化為高品質PPT](https://zhuanlan.zhihu.com/p/18105237248)
- 2025-01-07：[activepieces](https://github.com/activepieces/activepieces)：[一個開源的AI自動化工作流程工具](https://mp.weixin.qq.com/s/Z17KtGyAH5YI4R-VY1fgng)
- 2024-12-19：[LightRAG](https://github.com/HKUDS/LightRAG)；[LightRAG技術框架解讀](https://zhuanlan.zhihu.com/p/13261291813)
- 2024-12-15：[markitdown](https://github.com/microsoft/markitdown)


**[AI Agent 開源框架](https://deep-learning-101.github.io/agent)**
---

- [從AI Agent到Agent工作流程，一文詳細了解代理程式工作流程](https://zhuanlan.zhihu.com/p/32491596217)
- [萬字長文，帶你綜觀大模型Agent，涉及研究痛點、應用場景、發展方向](https://zhuanlan.zhihu.com/p/29833831482)
- [什麼是「Agentic 工作流程」？](https://zhuanlan.zhihu.com/p/32709535995)
- [什麼是Agentic AI？什麼是Agentic Workflow？與AI Agent有什麼區別和關聯？](https://zhuanlan.zhihu.com/p/705935464)
- [FinRobot](https://www.alphaxiv.org/zh/overview/2405.14767)；[DeepWiki](https://deepwiki.com/AI4Finance-Foundation/FinRobot)；可支援 Gemini-2.5-Pro-preview-05-06，基於 AutoGen    
- [Jupyter-AI](https://deepwiki.com/jupyterlab/jupyter-ai)；可支援 Gemini-2.5-Pro-preview-05-06

---

- 2025-11-15：[Agno](https://zread.ai/agno-agi/agno/)；[Agno 架構介紹：高性Multi-agent 系統框架深度解析](https://zhuanlan.zhihu.com/p/1945395802844410466)
- 2025-10-28：[Tongyi DeepResearch](https://zread.ai/Alibaba-NLP/DeepResearch)；[通義DeepResearch全面開源，超越openai deep research閉源框架](https://zhuanlan.zhihu.com/p/1951785880655209261)
- 2025-10-28：[DeepAgent](https://zread.ai/RUC-NLPIR/DeepAgent)；[DeepAgent: 首個全自主的深度推理智能體，可擴展大規模工具集](https://zhuanlan.zhihu.com/p/1966457879335798713)
- 2025-10-19：[Gemini Computer Use](https://ai.google.dev/gemini-api/docs/computer-use)；[Google推出Gemini 2.5 Computer Use讓AI代理能操作網頁介面](https://www.ithome.com.tw/news/171579)
- 2025-10-10：[SurfSense](https://zread.ai/MODSetter/SurfSense)；[GitHub 萬星新王炸，把你的Slack、Notion、Jira全餵給AI](https://mp.weixin.qq.com/s/za_ZQ7OWuvYaN2f0Ml0AgA)
- 2025-07-03：[multi-modal-researcher](https://github.com/langchain-ai/multi-modal-researcher)
- 2025-06-25：[Gemini CLI](https://github.com/google-gemini/gemini-cli)：[Gemini CLI：你的開源 AI 代理](https://blog.google/intl/zh-tw/products/cloud/gemini-cli-your-open-source-ai-agent/)
- 2025-06-06：[PandaWiki](https://github.com/chaitin/PandaWiki)；[新一代AI 大模型驅動的開源知識庫建立系統](https://zhuanlan.zhihu.com/p/1916981702733039060)
- 2025-06-03：[**Gemini Fullstack LangGraph**](https://deepwiki.com/google-gemini/gemini-fullstack-langgraph-quickstart)；[DEMO](https://deep-learning-101.github.io/gemini-fullstack-langgraph/FinGenAI)；[以為Google只是簡單放個Demo，哪想到是」開源版”Perplexity！](https://www.53ai.com/news/OpenSourceLLM/2025060431620.html)
- 2025-06-03：[Perplexica](https://github.com/ItzCrazyKns/Perplexica)；[Perplexity AI，開源替代品](https://www.53ai.com/news/qianyanjishu/2394.html)
- 2025-06-02：[Paper2Poster](https://paper2poster.github.io/)：[自動為論文產生海報](https://zhuanlan.zhihu.com/p/1912838595510776080)
- 2025-06-01：[**Agent Zero**](https://github.com/frdel/agent-zero)；[官網](https://agent-zero.ai/)；[這個自動AI代理可以做任何事！ （產生APP、程式碼、RAG 等）](https://cloud.tencent.com/developer/article/2472836)
- 2025-05-30：[WebDancer @ Alibaba](https://www.alphaxiv.org/zh/overview/2505.22648)；[DeepWiki](https://deepwiki.com/Alibaba-NLP/WebAgent)
- 2025-05-28：[**Lemon AI**](https://github.com/hexdocom/lemonai)；[全球首款全端開源通用AI Agent，讓人可以單機部署超級智慧體](https://www.53ai.com/news/OpenSourceLLM/2025052883904.html)
- 2025-05-25：[OpenHands](https://github.com/All-Hands-AI/OpenHands)；[Demo](https://app.all-hands.dev/)
- 2025-05-18：[Agent-Squad](https://deepwiki.com/awslabs/agent-squad)；[輕量級開源AI多智能體框架！智慧路由+上下文管理，前後端介面支援！](https://mp.weixin.qq.com/s/5Y23EhpHb2_pBOY8XrkMNw)
- 2025-05-10：[FlowGram](https://github.com/bytedance/flowgram.ai)：[字節跳動把Coze 核心開源了！視覺化工作流程引擎FlowGram 上線](https://mp.weixin.qq.com/s/EOtp8j67G5xd6H0qVfOhcw)；[如何使用](https://deepwiki.com/search/-dify-n8n_a61d08fd-2089-4cf3-9253-3275a54b54fa)
- 2025-05-10：[**DeerFlow**](https://github.com/bytedance/deer-flow/blob/main/README_zh.md)：[字節跳動DeerFlow深度解析](https://www.53ai.com/news/LargeLanguageModel/2025061552389.html)；[如何使用](https://deepwiki.com/search/_78a54d18-9132-44eb-920a-98618b505c9f)
- 2025-05-09：[**OpenDeepWiki**](https://github.com/AIDotNet/OpenDeepWiki)：[開源的DeekWiki加入MCP，輕鬆讓AI掌握開源專案使用文件！](https://mp.weixin.qq.com/s/Ux1-cpXdOSnjBrxCslHjtw)；[如何使用](https://deepwiki.com/search/_f9b90674-c6d9-4999-8a72-49cf28a30dca)
- 2025-05-07：[AI Manus](https://deepwiki.com/Simpleyyt/ai-manus)
- 2025-04-24：[suna](https://github.com/kortix-ai/suna)：[3週時間，就打造出Manus開源平替！貢獻原始碼，免費用](https://www.jiqizhixin.com/articles/2025-04-23-6)
- 2025-04-22：[釦子空間 (Coze Space)](https://space.coze.cn/)：[字節版Manus 釦子空間來了！實測效果絕佳，但還有3 個問題](https://zhuanlan.zhihu.com/p/1896900788091090915)
- 2025-04-03：[AutoAgent](https://github.com/HKUDS/AutoAgent)：[一句話全自動創建AI智能體，港大AutoAgent打造開源最強Deep Research](https://mp.weixin.qq.com/s/oATCuzI4BJ6JcwJkazinCA)
- 2025-04-03：[Agent Development Kit (ADK)](https://github.com/google/adk-python)：[谷歌發表「智能體開發工具包」ADK，來嚐個鮮](https://www.53ai.com/news/OpenSourceLLM/2025041012369.html)
- 2025-04-03：[**Deepsite**](https://huggingface.co/spaces/enzostvs/deepsite)；[DeepSite基於DeepSeek的網頁開發智能體，效果非常不錯](https://zhuanlan.zhihu.com/p/1890332067411243826)
- 2025-03-30：[DeepGemini](https://github.com/sligter/DeepGemini)：[AI界的'搭積木'神器，10分鐘打造你的專屬智慧團隊！](https://mp.weixin.qq.com/s/F2U7rWOMvfTyiRai-kfL_A)
- 2025-03-24：[**AgenticSeek**](https://github.com/Fosowl/agenticSeek)：[又一個“Manus”開源，完全本地化替代品AgenticSeek](https://mp.weixin.qq.com/s/tRZNgG2trzRxScP_fJ29JQ)
- 2025-03-20：[DeepSearcher](https://zread.ai/zilliztech/deep-searcher)；[DeepSearcher開源：告別傳統RAG，私有資料+Deepseek，打造本地版Deep Research](https://zhuanlan.zhihu.com/p/24273636289)
- 2025-03-11：[autoMate](https://github.com/yuruotong1/autoMate)：[autoMate:基於OmniParser 所建構的革命性AI自動化助手](https://mp.weixin.qq.com/s/7W0xL3EBJM9mmNZbdZCiiQ)
- 2025-03-10：[**OpenManus**](https://github.com/mannaandpoem/OpenManus)：[一文讀懂：OpenManus](https://zhuanlan.zhihu.com/p/30090038284)
- 2025-02-28：[**MoneyPrinterTurbo**](https://github.com/harry0703/MoneyPrinterTurbo)；[Al自動生成高清短視頻](https://zhuanlan.zhihu.com/p/27043978423)
- 2024-02-01：[MobileAgent](https://github.com/X-PLUG/MobileAgent/blob/main/README_zh.md)：[一句指示幫你操作手機，最新多模態手機助理Mobile-Agent來了！](https://zhuanlan.zhihu.com/p/680871671)
- 2025-01-03：[smolagents](https://github.com/huggingface/smolagents)：[新年禮物，Huggingface捲了一個Agent專案開源](https://blog.csdn.net/m0_59163425/article/details/144917058)
- 2024-09-23：[**STORM**](https://github.com/stanford-oval/storm)；[STORM：一个基于LLM的知识整理系统](https://mp.weixin.qq.com/s/x72eW958UbhrscvKghO6og)
- 2024-10-26：[OmniParser](https://github.com/microsoft/OmniParser)；[控制電腦手機的智慧體人人都能造，微軟開源OmniParser](https://zhuanlan.zhihu.com/p/3343331861)

**世界模型**

- 2025-09-25：[Code World Model](https://zread.ai/facebookresearch/cwm/1-overview)：[程式碼生成要變天了？被質疑架空後，Yann LeCun攜320億參數開源世界模型“殺回來了”](https://t.cj.sina.com.cn/articles/view/1746173800/68147f6801901e2wa)

**混合專家(Mixture of Experts, MoE)模型**

  - 2024-12-13：[DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)：[DeepSeek-VL2開源，VLM邁入MoE時代](https://mp.weixin.qq.com/s/s832KUgixNuX4GUkvY7_Ag)，[DeepSeek-VL2 先進視覺語言模型，在多模態理解方面取得了顯著進展](https://mp.weixin.qq.com/s/p6r_b-k4UnSJED5cBTedZg)
  - [騰訊混元](https://github.com/Tencent/Hunyuan-Large)：[騰訊混元又來開源，一出手就是最大MoE大模型](https://www.jiqizhixin.com/articles/2024-11-06-6)
     - 2024-11-06：[DEMO](https://huggingface.co/spaces/tencent/Hunyuan-Large)
     - 2024-11-06：[MODEL](https://huggingface.co/tencent/Hunyuan-Large)


**小型語言模型**

  - 2025-01-07：[Smolagents](https://github.com/huggingface/smolagents)：[Hugging Face開源全新AI智能體框架支援工具呼叫與程式碼執行！](https://zhuanlan.zhihu.com/p/16417392406)
  - 2024-12-13：[Phi-4](https://huggingface.co/NyxKrage/Microsoft_Phi-4)：[以小博大，微軟Phi-4正式發表~](https://mp.weixin.qq.com/s/uny1VUt7vk_ZU6hCH0EDGg)
  - 2024-11-18：[MobileLLM-1.5B](https://huggingface.co/facebook/MobileLLM-1.5B)：[Meta MobileLLM：深度架構與最佳化技術打造的行動裝置超強語言模型](https://mp.weixin.qq.com/s/hjY6L69pqN4GvybCuHesTA)
  - 2024-11-04：SmolLM2：[https://github.com/hiyouga/LLaMA-Factory](https://github.com/huggingface/smollm/)
     - [Hugging Face公布手機執行的小型語言模型SmolLM2](https://www.ithome.com.tw/news/165832)
  - 2024-09-25：[Llama 3.2 90b, 11b, 3b, 1b: Revolutionizing edge AI and vision with open, customizable models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

**推理模型**

  - 2025-08-05：[gpt-oss](https://huggingface.co/openai/gpt-oss-120b)；[隆重介紹 gpt-oss](https://openai.com/zh-Hant/index/introducing-gpt-oss/)；[OpenAI重新開源！深夜連發兩個推理模型，o4-mini水平](https://www.jiqizhixin.com/articles/2025-08-06-2)
  - 2025-07-29：[Llama Nemotron Super v1.5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)；[英偉達全新開源模型：三倍吞吐、單卡可跑，還拿下推理SOTA](https://zhuanlan.zhihu.com/p/1933514869279274584)
  - 2025-07-27：[OpenReasoning-Nemotron](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)；[英偉達突然開源「數學核武」！ 1.5B 參數秒殺 o3，OpenReasoning-Nemotron 真有這麼猛？](https://mp.weixin.qq.com/s/o7RhRAFzAKkHj2T0y3GVzA)
  - 2025-05-06：[Llama-Nemotron](https://www.alphaxiv.org/zh/overview/2505.00949)；[DeepWiki](https://deepwiki.com/NVIDIA/NeMo)；[英偉達發布Llama-Nemotron系列大模型，實現高效推理](https://zhuanlan.zhihu.com/p/1903012593033012833)
  - 2025-04-16：[Video-R1: Reinforcing Video Reasoning in MLLMs ](https://www.alphaxiv.org/zh/overview/2503.21776)；[Github](https://github.com/tulerfeng/Video-R1)；[影片推理R1時刻，7B模型反超GPT-4o，港中文清華推出首款Video-R1](https://www.36kr.com/p/3252742390655489)

    
**大型語言模型**
- 2025-08-05：[Claude Opus 4.1](https://www.jiqizhixin.com/articles/2025-08-06-4)
- 2024-11-23：[Ai2 Tülu 3](https://github.com/allenai/open-instruct)：[這才是真・開源模型！公開「後訓練」一切，性能超越Llama 3.1 Instruct](https://www.jiqizhixin.com/articles/2024-11-23-5)
  - DEMO：[https://playground.allenai.org/](https://playground.allenai.org/)
  - MODEL：[https://huggingface.co/allenai](https://huggingface.co/allenai)
- 2024-11-9：[Ai2 OpenScholar](https://allenai.org/blog/openscholar)：[https://openscholar.allen.ai/](https://openscholar.allen.ai/)
- 2024-09-25：[Llama 3.2 90b, 11b, 3b, 1b: Revolutionizing edge AI and vision with open, customizable models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)


**Embedding & Reranker**
- 2025-07-14：[gemini-embedding-001	](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings?hl=zh-tw)
- 2025-06-05：[Qwen3 Embedding：新一代文本表徵與排序模型](https://qwenlm.github.io/zh/blog/qwen3-embedding/)
   - 2025-06-03：[Qwen3-Embedding](https://huggingface.co/collections/Qwen/qwen3-embedding-6841b2055b99c44d9a4c371f)
   - 2025-06-03：[Qwen3-Reranker](https://huggingface.co/collections/Qwen/qwen3-reranker-6841b22d0192d7ade9cdefea)

**語音到語音大模型及工具套件**
- [ten-agent](https://github.com/TEN-framework/TEN-Agent)
   - [又一款王炸級的開源端對端語音模型](https://mp.weixin.qq.com/s/pw9LQyRCRogfxAlYG3EfcQ)
   - [TEN Framework 入坑记](https://mp.weixin.qq.com/s/ZVZHNP0XPwzGapWWqTk1kw)
   - [使用 TEN Agent 搭建 Conversation AI Agent](https://uy6npdpeoi.feishu.cn/docx/EAWYdWWO7ormNPxUhyVcO3GSnUc)
- [pipecat](https://github.com/pipecat-ai/pipecat)：[OpenAI工程師親自修訂：用ChatGPT即時語音API建立應用](https://www.jiqizhixin.com/articles/2025-01-10-4)
- 2025-11-03：[LongCat-Flash-Omni](https://huggingface.co/meituan-longcat/LongCat-Flash-Omni)；[LongCat-Flash-Omni正式發布並開源：開啟全模態即時互動時代](https://zhuanlan.zhihu.com/p/1968699530762491165)
- 2025-07-21：[Audio Flamingo 3 @ NVIDIA](https://github.com/NVIDIA/audio-flamingo)；[NVIDIA 開源多模態音訊模型Audio Flamingo 3](https://www.oschina.net/news/361477/nvidia-audio-flamingo-3)
- 2025-05-08：[Voila](https://github.com/maitrix-org/Voila)；[新型開源端對端AI 語音模型！ Voila：195ms 超低延遲引領全雙工對話！](https://zhuanlan.zhihu.com/p/1903776373765547954)
- [HuggingFace Speech-to-Speech](https://github.com/huggingface/speech-to-speech)

    
**視覺大語言模型 (Vision-Language model)**
- 2025-05-20：[Seed1.5-VL](https://github.com/ByteDance-Seed/Seed1.5-VL)；[Seed1.5-VL：具有視覺增強多模態能力的高階語言模型](https://www.alphaxiv.org/zh/overview/2505.07062)；[字節跳動發布Seed1.5-VL視覺-語言多模態大模型，實測效果非常不錯](https://zhuanlan.zhihu.com/p/1905914968433497765)
- 2025-05-12：[nanoVLM](https://deepwiki.com/huggingface/nanoVLM)


**多模態大語言模型 (Multimodal)**
- [InternVL](https://github.com/OpenGVLab/InternVL)
   - [(CVPR 2024 Oral) InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://www.alphaxiv.org/zh/overview/2504.10479)
   - [DeepWiki](https://deepwiki.com/OpenGVLab/InternVL)；[InternVL3：刷新開源多模態大模型效能新紀錄](https://zhuanlan.zhihu.com/p/1897681159359551408)
- 2025-05-24：[Dolphin](https://www.alphaxiv.org/zh/overview/2505.14059)：[DeepWiki](https://deepwiki.com/bytedance/Dolphin)；[開源多模態複雜文件解析模型！ Dolphin](https://zhuanlan.zhihu.com/p/1911355829485045020)
- 2025-05-21：[Gemma 3n](https://deepmind.google/models/gemma/?hl=zh-tw)；[Preview](https://huggingface.co/google/gemma-3n-E4B-it-litert-preview)  
- 2025-03-18：[Mistral Small 3.1](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)：[Mistral開源多模態小模型3.1：128K上下文+超低延遲，效能碾壓GPT-4o Mini](https://zhuanlan.zhihu.com/p/31138756743)
- 2025-03-14：[Vision-R1](https://github.com/Osilly/Vision-R1)：[Vision-R1：激發多模態大模型的推理能力](https://zhuanlan.zhihu.com/p/29618155786)
- 2025-02-28：[HumanOmni](https://github.com/HumanMLLM/HumanOmni)
   - [阿里通義開源業界首個第一視角大模型，超強的視訊理解效能！](https://mp.weixin.qq.com/s/acn16cvE8N4tMegKuGHAKQ)
   - [首個專注於人類中心場景的多模態大模型，視覺與聽覺融合的突破！](https://mp.weixin.qq.com/s/cO6xEAOCRUsLmoiDbq12tw)
- [Phi](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4)
   - [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
   - 2025-02-27
      - [微軟首個多模態Phi-4問世，56億參數秒殺GPT-4o！ LoRA華人大佬帶隊](https://zhuanlan.zhihu.com/p/26984226500)
      - [微軟發表Phi-4-Mini系列模型：小身材，大智慧－語言與多模態AI的新突破](https://zhuanlan.zhihu.com/p/26678433652)
   - 2024-09-12：[微軟AI發布Phi 3.5 mini、MoE 和Vision](https://mp.weixin.qq.com/s/EeALIBrvGWKtEBGnroZIvg)  
- [MiniCPM](https://github.com/OpenBMB)
   - 2025-01-16：[MiniCPM-o 2.6：流式全模態，端到端，多模態端側大模型來了！](https://mp.weixin.qq.com/s/bTRirDr-MCscYF88KmK5qw)；[文件](https://github.com/OpenBMB/MiniCPM-o/blob/main/README_zh.md#minicpm-o-26)
  - 2024-09-11：[升級Ollama！ MiniCPM-V2_6影像辨識模型上線](https://mp.weixin.qq.com/s/6N-u8PcGEX6e4rryeqXglQ)
   - 2024-09-06：[MiniCPM 3.0 開源！ 4B參數超GPT3.5性能，無限長文本，超強RAG三件套！模型推理、微調實戰來啦！](https://53ai.com/news/OpenSourceLLM/2024090659871.html)：[https://github.com/OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)
  - 2024-09-05：[零碼基礎都敢去魔改MiniCPM-V了？是我飄了，也是Cursor 太強了](https://mp.weixin.qq.com/s/DjDznmtKZoJNKXYz0X4zog)：[https://github.com/OpenBMB/MiniCPM-V/](https://github.com/OpenBMB/MiniCPM-V/)

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/Large-Language-Model"
  },
  "headline": "大語言模型 (Large Language Model) 資源彙整",
  "description": "一份詳盡的大語言模型（LLM）資源清單，涵蓋模型排行榜、中文LLM資源、微調技術、開源工具、AI Agent 框架以及最新的模型發布，由台灣深度學習同好會（Deep Learning 101）提供。",
  "image": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg", // 建議您換成一個代表性的圖片 URL
  "author": {
    "@type": "Organization",
    "name": "Deep Learning 101, Taiwan",
    "url": "https://deep-learning-101.github.io/"
  },
  "publisher": {
    "@type": "Organization",
    "name": "Deep Learning 101, Taiwan",
    "logo": {
      "@type": "ImageObject",
      "url": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg" // 建議您換成 LOGO 圖片 URL
    }
  },
  "datePublished": "2016-11-11", // 根據 front matter 的資訊，您也可以更新為內容的最後更新日期
  "dateModified": "2025-10-21", // 建議填寫您最後修改此文件的日期
  "keywords": "Large Language Model, LLM, AI Agent, Fine-tuning, RAG, Deep Learning, 生成式AI, 大語言模型, 微調, 開源工具",
  "about": {
    "@type": "Service",
    "serviceType": "GenAI Consulting",
    "provider": {
      "@type": "Organization",
      "name": "Deep Learning 101, Taiwan"
    },
    "name": "生成式 AI 諮詢 (GenAI Consulting)",
    "description": "提供關於大語言模型（LLM）的專業諮詢服務，包含模型微調、應用開發、框架選擇與技術導入。"
  }
}
</script>