---
layout: default
title: LLM 大語言模型資源懶人包 | Leaderboard, Tools & Papers | Deep Learning 101
description: 2025 最新 LLM 大語言模型資源彙整。包含 Open LLM Leaderboard、Ollama/vLLM 部署工具、Llama 3/Mistral 等必讀論文，以及中文 LLM 微調資源。
permalink: /Large-Language-Model
lang: zh-Hant
schema_type: article
---

{% include header.html %}

# 📚 LLM 大語言模型・必讀資源總整理

> **編者按：** 本頁面彙整目前最主流的 LLM 排行榜、開源模型、推論與微調工具，以及相關學術論文。
>
> 如果您想尋找更詳細的筆記，歡迎訪問 **GitHub Repository**：
> 👉 [**GitHub: Natural-Language-Processing-Paper**](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper) (歡迎 Star ⭐)

---

{% include ai-share.html %}

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

---

{% include price.html %}

---

### **文章目錄**
- [🏆 排行榜 (Leaderboards)](#leaderboards)
- [🛠️ 微調技術與資源 (Fine-tuning)](#fine-tuning)
- [🧩 AI Agent 開源框架](#ai-agent)
- [🛠️ 開發工具 (Tools & Protocols)](#tools)
- [🌍 World Models (世界模型)](#world-models)
- [🧠 MoE (混合專家模型)](#moe)
- [📱 Small Language Models (小型語言模型)](#slm)
- [🤔 Reasoning Models (推理模型)](#reasoning)
- [🏛️ Large Language Models (大型語言模型)](#llm)
- [🔎 Embedding & Reranker](#embedding)
- [🔊 Speech-to-Speech LLM (語音大模型)](#speech)
- [👁️ Vision-Language Model (視覺大語言模型)](#vision)
- [🌌 Multimodal LLM (多模態大語言模型)](#multimodal)

---

## Leaderboards
**🏆 排行榜 (Leaderboards)**

- [**AlpacaEval Leaderboard**](https://tatsu-lab.github.io/alpaca_eval/)
- [**Open LLM Leaderboard**](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [**Big Code Models Leaderboard**](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
- [**Awesome-Chinese-LLM**](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)

---

## Fine-tuning
**🛠️ 微調技術與資源 (Fine-tuning)**

### 顯存估算 (VRAM)
- **大模型所需 GPU 記憶體筆記**
  - 資源：[📝 微信公眾號](https://mp.weixin.qq.com/s/M_hdtR7mVq14MnaaL0MAUw)
- **不同參數規模在微調方法下所需的顯存總結**
  - 資源：[📝 DataLearner](https://www.datalearner.com/blog/1051703254378255)

### 微調技術與教學
- **微調技術全解**
  - 說明：SFT、LoRA、P-tuning v2、Freeze 監督微調方法
  - 資源：[📝 知乎專欄](https://zhuanlan.zhihu.com/p/643941480)
- **LoRA vs 完全微調**
  - 說明：MIT 21頁論文講懂了
  - 資源：[📝 機器之心](https://www.jiqizhixin.com/articles/2024-11-11-5)
- **大模型微調 (Fine-tuning) 全解**
  - 資源：[📝 53AI](https://www.53ai.com/news/finetuning/2025022604125.html)
- **Unsloth 官方微調技巧**
  - 說明：初學者必看指南
  - 資源：[📝 微信公眾號](https://mp.weixin.qq.com/s/COZfH_h36nX33TZGBVn0rg)
- **零代碼一站式微調**
  - 說明：從資料集準備到模型微調全流程
  - 資源：[📝 知乎專欄](https://zhuanlan.zhihu.com/p/1906670241645322809)
- **DeepSeek-R1 微調指南**
  - 說明：微調為領域專家
  - 資源：[📝 知乎專欄](https://zhuanlan.zhihu.com/p/25054526736)
- **NVIDIA NeMo**
  - 說明：模型剪枝和知識蒸餾
  - 資源：[📝 NVIDIA Blog](https://developer.nvidia.com/zh-cn/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/)

### 微調框架 (Frameworks)
- **LLaMA Factory**
  - 資源：[🐙 GitHub](https://github.com/hiyouga/LLaMA-Factory) | [🤗 Demo](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
  - 延伸：[📝 中文文檔](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md) | [📝 架構解析 (2024-09-13)](https://mp.weixin.qq.com/s/eJqKc_2nHBYzDFAp2AYdWQ) | [📝 單卡訓練 Agent 實戰](https://zhuanlan.zhihu.com/p/678989191)

- **Torchtune**
  - 資源：[🐙 GitHub](https://github.com/pytorch/torchtune) | [📖 官方文件](https://pytorch.dev.org.tw/torchtune/stable/index.html)
  - 延伸：[📝 Llama3.1 知識蒸餾實戰](https://pytorch.ac.cn/torchtune/stable/tutorials/llama_kd_tutorial.html)

### 資料集準備 (Datasets)
- **微調資料集實戰**
  - 資源：[📝 資料集怎麼搞？](https://zhuanlan.zhihu.com/p/29522986573) | [📝 LLaMA Factory 資料集建立](https://zhuanlan.zhihu.com/p/1916489160333714285)
- **Easy Dataset**
  - 說明：大模型微調資料集生產工具
  - 資源：[📝 知乎專欄](https://zhuanlan.zhihu.com/p/1908313086064042177)
- **OpenDeepWiki**
  - 說明：根據現有檔案產生微調資料集
  - 資源：[📝 知乎專欄](https://zhuanlan.zhihu.com/p/1908831694879985815)
- **COIG-CQIA**
  - 說明：零一萬物發布高品質中文指令微調數據
  - 資源：[📝 知乎專欄](https://zhuanlan.zhihu.com/p/694434197)

---

## AI-Agent
**🧩 AI Agent 開源框架**
> 完整列表請見：[避開 AI 代理 (AI Agents) 與 代理式人工智慧 (Agentic AI) 開發陷阱](https://deep-learning-101.github.io/agent)

### 核心概念與必讀文章
- **AI Search Has A Citation Problem**
  - 資源：[📝 CJR](https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php)
- **Agentic AI vs AI Agents**
  - 說明：A Conceptual Taxonomy, Applications and Challenges
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2505.10468)
- **OWASP Agentic AI**
  - 說明：Threats and Mitigations
  - 資源：[🛡️ OWASP](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/)
- **Agent 工作流入門**
  - 資源：[📝 從 Agent 到 Workflow](https://zhuanlan.zhihu.com/p/32491596217) | [📝 萬字長文綜觀 Agent](https://zhuanlan.zhihu.com/p/29833831482) | [📝 什麼是 Agentic 工作流程？](https://zhuanlan.zhihu.com/p/32709535995) | [📝 Agentic AI 區別](https://zhuanlan.zhihu.com/p/705935464)
- **FinRobot**
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2405.14767) | [📚 DeepWiki](https://deepwiki.com/AI4Finance-Foundation/FinRobot) (支援 Gemini 2.5)
- **Jupyter-AI**
  - 資源：[📚 DeepWiki](https://deepwiki.com/jupyterlab/jupyter-ai) (支援 Gemini 2.5)

### Agent 框架列表 (按時間排序)

- 2025-11-15｜**Agno**
  - 說明：高效能 Multi-agent 系統框架
  - 資源：[🌐 官網](https://zread.ai/agno-agi/agno/) | [📝 架構深度解析](https://zhuanlan.zhihu.com/p/1945395802844410466)

- 2025-10-28｜**Tongyi DeepResearch**
  - 說明：通義全面開源，超越 OpenAI 閉源框架
  - 資源：[📝 DeepResearch](https://zread.ai/Alibaba-NLP/DeepResearch) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1951785880655209261)

- 2025-10-28｜**DeepAgent**
  - 說明：首個全自主深度推理智能體
  - 資源：[📝 RUC-NLPIR](https://zread.ai/RUC-NLPIR/DeepAgent) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1966457879335798713)

- 2025-10-19｜**Gemini Computer Use**
  - 說明：Google 推出讓 AI 代理操作網頁介面
  - 資源：[📖 官方文件](https://ai.google.dev/gemini-api/docs/computer-use) | [📝 iThome 報導](https://www.ithome.com.tw/news/171579) | [🐙 GitHub](https://github.com/google-gemini/computer-use-preview)

- 2025-10-10｜**SurfSense**
  - 說明：GitHub 萬星王炸，整合 Slack/Notion/Jira
  - 資源：[📝 MODSetter](https://zread.ai/MODSetter/SurfSense) | [📝 中文解讀](https://mp.weixin.qq.com/s/za_ZQ7OWuvYaN2f0Ml0AgA)

- 2025-08-29｜**Microsoft Agent Framework**
  - 說明：開放原始碼開發套件，用於建置 .NET 和 Python 的 AI 代理程式 和 多代理程式工作流程 。
  - 資源：[🐙 GitHub](https://github.com/microsoft/agent-framework) | [📝 官方文件](https://learn.microsoft.com/zh-tw/agent-framework/overview/agent-framework-overview)

- 2025-07-03｜**multi-modal-researcher**
  - 資源：[🐙 GitHub](https://github.com/langchain-ai/multi-modal-researcher)

- 2025-06-25｜**Gemini CLI**
  - 說明：你的開源 AI 代理
  - 資源：[🐙 GitHub](https://github.com/google-gemini/gemini-cli) | [📝 Google Blog](https://blog.google/intl/zh-tw/products/cloud/gemini-cli-your-open-source-ai-agent/)

- 2025-06-06｜**PandaWiki**
  - 說明：新一代 AI 大模型驅動的開源知識庫
  - 資源：[🐙 GitHub](https://github.com/chaitin/PandaWiki) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1916981702733039060)

- 2025-06-03｜**Gemini Fullstack LangGraph**
  - 說明：開源版 Perplexity
  - 資源：[📚 DeepWiki](https://deepwiki.com/google-gemini/gemini-fullstack-langgraph-quickstart) | [🌐 DEMO](https://deep-learning-101.github.io/gemini-fullstack-langgraph/FinGenAI) | [📝 53AI 報導](https://www.53ai.com/news/OpenSourceLLM/2025060431620.html)

- 2025-06-03｜**Perplexica**
  - 說明：Perplexity AI 開源替代品
  - 資源：[🐙 GitHub](https://github.com/ItzCrazyKns/Perplexica) | [📝 53AI 報導](https://www.53ai.com/news/qianyanjishu/2394.html)

- 2025-06-02｜**Paper2Poster**
  - 說明：自動為論文產生海報
  - 資源：[🌐 Project](https://paper2poster.github.io/) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1912838595510776080)

- 2025-06-01｜**Agent Zero**
  - 說明：全能 AI 代理（產生APP、程式碼、RAG）
  - 資源：[🐙 GitHub](https://github.com/frdel/agent-zero) | [🌐 官網](https://agent-zero.ai/) | [📝 騰訊雲文章](https://cloud.tencent.com/developer/article/2472836)

- 2025-05-30｜**WebDancer**
  - 說明：Alibaba 開源 WebAgent
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.22648) | [📚 DeepWiki](https://deepwiki.com/Alibaba-NLP/WebAgent)

- 2025-05-28｜**Lemon AI**
  - 說明：全球首款全端開源通用 AI Agent
  - 資源：[🐙 GitHub](https://github.com/hexdocom/lemonai) | [📝 53AI 報導](https://www.53ai.com/news/OpenSourceLLM/2025052883904.html)

- 2025-05-25｜**OpenHands**
  - 資源：[🐙 GitHub](https://github.com/All-Hands-AI/OpenHands) | [🌐 Demo](https://app.all-hands.dev/)

- 2025-05-18｜**Agent-Squad**
  - 說明：輕量級開源 AI 多智能體框架 (AWS Labs)
  - 資源：[📚 DeepWiki](https://deepwiki.com/awslabs/agent-squad) | [📝 中文解讀](https://mp.weixin.qq.com/s/5Y23EhpHb2_pBOY8XrkMNw)

- 2025-05-10｜**FlowGram (ByteDance)**
  - 說明：字節跳動開源 Coze 核心工作流引擎
  - 資源：[🐙 GitHub](https://github.com/bytedance/flowgram.ai) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/EOtp8j67G5xd6H0qVfOhcw) | [📚 DeepWiki](https://deepwiki.com/search/-dify-n8n_a61d08fd-2089-4cf3-9253-3275a54b54fa)

- 2025-05-10｜**DeerFlow**
  - 說明：字節跳動 DeerFlow 解析
  - 資源：[🐙 GitHub](https://github.com/bytedance/deer-flow/blob/main/README_zh.md) | [📝 深度解析](https://www.53ai.com/news/LargeLanguageModel/2025061552389.html) | [📚 DeepWiki](https://deepwiki.com/search/_78a54d18-9132-44eb-920a-98618b505c9f)

- 2025-05-09｜**OpenDeepWiki**
  - 說明：加入 MCP，讓 AI 掌握開源專案文件
  - 資源：[🐙 GitHub](https://github.com/AIDotNet/OpenDeepWiki) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/Ux1-cpXdOSnjBrxCslHjtw) | [📚 如何使用](https://deepwiki.com/search/_f9b90674-c6d9-4999-8a72-49cf28a30dca)

- 2025-05-07｜**AI Manus**
  - 資源：[📚 DeepWiki](https://deepwiki.com/Simpleyyt/ai-manus)

- 2025-04-24｜**suna**
  - 說明：Manus 開源平替
  - 資源：[🐙 GitHub](https://github.com/kortix-ai/suna) | [📝 機器之心](https://www.jiqizhixin.com/articles/2025-04-23-6)

- 2025-04-22｜**釦子空間 (Coze Space)**
  - 說明：字節版 Manus
  - 資源：[🌐 官網](https://space.coze.cn/) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1896900788091090915)

- 2025-04-03｜**AutoAgent**
  - 說明：港大打造開源最強 Deep Research
  - 資源：[🐙 GitHub](https://github.com/HKUDS/AutoAgent) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/oATCuzI4BJ6JcwJkazinCA)

- 2025-04-03｜**Agent Development Kit (ADK)**
  - 說明：Google 智能體開發工具包
  - 資源：[🐙 GitHub](https://github.com/google/adk-python) | [📝 53AI 報導](https://www.53ai.com/news/OpenSourceLLM/2025041012369.html)

- 2025-04-03｜**Deepsite**
  - 說明：基於 DeepSeek 的網頁開發智能體
  - 資源：[🤗 Space](https://huggingface.co/spaces/enzostvs/deepsite) | [📝 知乎推薦](https://zhuanlan.zhihu.com/p/1890332067411243826)

- 2025-03-30｜**DeepGemini**
  - 說明：AI 界搭積木神器
  - 資源：[🐙 GitHub](https://github.com/sligter/DeepGemini) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/F2U7rWOMvfTyiRai-kfL_A)

- 2025-03-24｜**AgenticSeek**
  - 說明：Manus 完全本地化替代品
  - 資源：[🐙 GitHub](https://github.com/Fosowl/agenticSeek) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/tRZNgG2trzRxScP_fJ29JQ)

- 2025-03-20｜**DeepSearcher**
  - 說明：私有資料 + Deepseek 打造本地 Deep Research
  - 資源：[📝 DeepSearcher](https://zread.ai/zilliztech/deep-searcher) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/24273636289)

- 2025-03-11｜**autoMate**
  - 說明：基於 OmniParser 的 AI 自動化助手
  - 資源：[🐙 GitHub](https://github.com/yuruotong1/autoMate) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/7W0xL3EBJM9mmNZbdZCiiQ)

- 2025-03-10｜**OpenManus**
  - 資源：[🐙 GitHub](https://github.com/mannaandpoem/OpenManus) | [📝 一文讀懂](https://zhuanlan.zhihu.com/p/30090038284)

- 2025-02-28｜**MoneyPrinterTurbo**
  - 說明：AI 自動生成高清短視頻
  - 資源：[🐙 GitHub](https://github.com/harry0703/MoneyPrinterTurbo) | [📝 知乎推薦](https://zhuanlan.zhihu.com/p/27043978423)

- 2024-02-01｜**MobileAgent**
  - 說明：多模態手機助理
  - 資源：[🐙 GitHub](https://github.com/X-PLUG/MobileAgent/blob/main/README_zh.md) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/680871671)

- 2025-01-03｜**smolagents**
  - 說明：Hugging Face 開源 Agent 框架
  - 資源：[🐙 GitHub](https://github.com/huggingface/smolagents) | [📝 CSDN 介紹](https://blog.csdn.net/m0_59163425/article/details/144917058)

- 2024-10-26｜**OmniParser**
  - 說明：微軟開源，控制電腦手機的智能體
  - 資源：[🐙 GitHub](https://github.com/microsoft/OmniParser) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/3343331861)

- 2024-09-23｜**STORM**
  - 說明：基於 LLM 的知識整理系統 (Stanford)
  - 資源：[🐙 GitHub](https://github.com/stanford-oval/storm) | [📝 公眾號介紹](https://mp.weixin.qq.com/s/x72eW958UbhrscvKghO6og)

---

## Tools
**🛠️ 開發工具 (Tools & Protocols)**

### MCP (Model Context Protocol)
- 2025-08-20｜**FastAPI-MCP**
  - 說明：將 FastAPI 介面升級為 MCP 工具服務
  - 資源：[📝 zread](https://zread.ai/tadata-org/fastapi_mcp) | [📝 公眾號教學](https://mp.weixin.qq.com/s/L568EP2tl2zwmC8vxz8s7w)
- 2025-04-15｜**automcp**
  - 說明：秒設定 MCP 伺服器
  - 資源：[🐙 GitHub](https://github.com/NapthaAI/automcp) | [📝 公眾號介紹](https://mp.weixin.qq.com/s/x-aZEhtnRYPFno81Fb9ttw)
- 2025-04-10｜**line-bot-mcp-server**
  - 資源：[🐙 GitHub](https://github.com/line/line-bot-mcp-server)
- 2025-04-05｜**GitMCP**
  - 說明：讓 AI 秒懂 GitHub 項目
  - 資源：[🐙 GitHub](https://github.com/idosal/git-mcp) | [📝 53AI 報導](https://www.53ai.com/news/RAG/2025040590146.html)
- 2025-03-14｜**playwright-mcp**
  - 說明：AI 自動化神器
  - 資源：[🐙 GitHub](https://github.com/microsoft/playwright-mcp) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/30178146112)

### Browser Automation (瀏覽器自動化)
- **Browser-use**
  - 資源：[🐙 GitHub](https://github.com/browser-use/browser-use)
  - 2025-06-04：[workflow-use](https://github.com/browser-use/workflow-use) (一次錄製，永久使用)
  - 2025-04-16：[web-ui](https://github.com/browser-use/web-ui) | [📚 如何使用](https://deepwiki.com/search/_bfd33aa8-cd79-4f1d-a1e8-5620d4374329)
  - 2025-03-28：[browser-use-webui](https://github.com/browser-use/web-ui)
  - 2025-02-16：[webui 部署教學](https://zhuanlan.zhihu.com/p/24116360552)
  - 2025-01-23：[讓 AI 像人類一樣使用瀏覽器](https://zhuanlan.zhihu.com/p/20038156945)

### 效率工具 (Efficiency Tools)
- 2025-12-20｜**NVIDIA Nemotron-3-Nano**
  - 資源：[🤗 HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) | [🌐 OpenRouter](https://openrouter.ai/nvidia/nemotron-3-nano-30b-a3b:free)
- 2025-11-20｜**LinearRAG**
  - 說明：全新 RAG 框架，無需關係抽取
  - 資源：[🐙 GitHub](https://github.com/DEEP-PolyU/LinearRAG) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1975321777342260763)
- 2025-09-11｜**DeepMCPAgent**
  - 說明：教你讓模型自己「找工具」
  - 資源：[📝 zread](https://zread.ai/cryxnet/DeepMCPAgent) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/Sj_7i1mTJ9WYaTlCzIqCFA)
- 2025-07-30｜**LangExtract**
  - 說明：Gemini 驅動的資訊擷取庫
  - 資源：[🐙 GitHub](https://github.com/google/langextract) | [📝 Google Developers](https://developers.googleblog.com/zh-hans/introducing-langextract-a-gemini-powered-information-extraction-library/)
- 2025-06-28｜**docext**
  - 說明：基於 Qwen2.5VL 的文檔解析工具
  - 資源：[🐙 GitHub](https://github.com/NanoNets/docext) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1919760450024879687)
- 2025-06-10｜**Agentic-Doc**
  - 說明：LandingAI 開源，百頁文檔秒變結構化資料
  - 資源：[🐙 GitHub](https://github.com/landing-ai/agentic-doc) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1914259475306612709)
- 2025-06-06｜**daily-arXiv-ai-enhanced**
  - 說明：每日爬取 arXiv 並用 LLM 產生中文摘要
  - 資源：[🐙 GitHub](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced)
- 2025-05-22｜**AingDesk**
  - 說明：零門檻本地 AI 部署
  - 資源：[📚 DeepWiki](https://deepwiki.com/aingdesk/AingDesk) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/29773848356)
- 2025-05-20｜**news-agents**
  - 資源：[📚 DeepWiki](https://deepwiki.com/eugeneyan/news-agents)
- 2025-05-16｜**Follow**
  - 說明：資訊聚合神器
  - 資源：[📚 DeepWiki](https://deepwiki.com/RSSNext/Folo) | [📝 知乎推薦](https://zhuanlan.zhihu.com/p/1906505020628795653)
- 2025-05-11｜**SurfSense**
  - 說明：打通 Notion/GitHub 的 AI 超腦
  - 資源：[🐙 GitHub](https://github.com/MODSetter/SurfSense) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/kMhidgb6GkKEsl-D-u_7iw) | [📚 如何使用](https://deepwiki.com/search/_df4a192b-a253-4155-a2a2-4a6fda9037e9)
- 2025-04-28｜**PaperCoder (Paper2Code)**
  - 說明：Automating Code Generation from Scientific Papers
  - 資源：[📚 DeepWiki](https://deepwiki.com/going-doer/Paper2Code) | [📄 AlphaXiv](https://www.alphaxiv.org/overview/2504.17192)
- 2025-04-16｜**OneFileLLM**
  - 說明：一鍵聚合網頁、程式碼、論文到剪貼簿
  - 資源：[🐙 GitHub](https://github.com/jimmc414/onefilellm) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/qNYX65fw-IWzEBLZpuaY6Q)
- 2025-04-16｜**ScrapeGraphAI**
  - 說明：自然語言驅動的智慧爬蟲
  - 資源：[🐙 GitHub](https://github.com/ScrapeGraphAI/Scrapegraph-ai) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/lQukAy12V5K1cH6rTkqxaA)
- 2025-04-15｜**stagehand**
  - 說明：AI 驅動的下一代瀏覽器自動化框架
  - 資源：[🐙 GitHub](https://github.com/browserbase/stagehand) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/KF-z67kn4rTjcIBmTvj3nA)
- 2025-04-11｜**nanobrowser**
  - 說明：AI 驅動的瀏覽器自動化神器
  - 資源：[🐙 GitHub](https://github.com/nanobrowser/nanobrowser) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/65SwCtDta1cKvx1_BbaoHQ)
- 2025-04-10｜**DevDocs**
  - 說明：開發者的文檔收割機
  - 資源：[🐙 GitHub](https://github.com/cyberagiinc/DevDocs) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/k5fG_L1q_19ylKIJD6PXmw)
- 2025-04-06｜**sqlchat**
  - 說明：讓資料庫管理像聊天一樣簡單
  - 資源：[🐙 GitHub](https://github.com/sqlchat/sqlchat) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/kieSzWn3QDYvZ5Zx35hr1A)
- 2025-03-26｜**pdf-craft**
  - 說明：PDF 秒轉 Markdown/EPUB
  - 資源：[🐙 GitHub](https://github.com/oomol-lab/pdf-craft) | [📝 知乎推薦](https://zhuanlan.zhihu.com/p/1888288260171744707)
- 2025-03-25｜**OCRmyPDF**
  - 說明：能力分析
  - 資源：[🐙 GitHub](https://github.com/ocrmypdf/OCRmyPDF) | [📝 知乎分析](https://www.zhihu.com/tardis/zm/art/32745781279?source_id=1003)
- 2025-03-12｜**AingDesk** (同上)
  - 資源：[📚 DeepWiki](https://deepwiki.com/aingdesk/AingDesk) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/29773848356)
- 2025-03-08｜**composio**
  - 說明：AI 助理效率神器，整合 200+ 工具
  - 資源：[🐙 GitHub](https://github.com/ComposioHQ/composio) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/rRPOmihGzcIXx0HQc3pdoA)
- 2025-02-25｜**PySpur**
  - 說明：拖曳式開發 AI 工作流程
  - 資源：[🌐 官網](https://www.pyspur.dev/) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/26161709083)
- 2025-01-13｜**DocAligner**
  - 說明：拍照文件復原 (校正、版面定位)
  - 資源：[🐙 GitHub](https://github.com/ZZZHANG-jx/DocAligner) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/Bra9h3ExddB5NiH1g4uk1g)
- 2025-01-07｜**activepieces**
  - 說明：開源 AI 自動化工作流程工具
  - 資源：[🐙 GitHub](https://github.com/activepieces/activepieces) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/Z17KtGyAH5YI4R-VY1fgng)
- 2024-12-19｜**LightRAG**
  - 資源：[🐙 GitHub](https://github.com/HKUDS/LightRAG) | [📝 技術框架解讀](https://zhuanlan.zhihu.com/p/13261291813)
- 2024-12-15｜**markitdown**
  - 資源：[🐙 GitHub](https://github.com/microsoft/markitdown)

---

## AI PTT
**🌍 AI PPT (用AI做PPT)**

-2026-01-04 | **LangChat Slides**
  - 說明：基於生成式AI 的智慧幻燈片生成工具，由LangChat 團隊開發。
  - 資源：[🐙 GitHub](https://github.com/langchat/langchat-slides ) | [DEMO](https://slides.langchat.cn/) | [掘金解讀](https://juejin.cn/post/7591861857465778214)
- 2025-07-26｜**presenton**
  - 說明：本地部署一鍵生成精美 PPT
  - 資源：[🐙 GitHub](https://github.com/presenton/presenton) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/QTMVGD_aP41qrwtbjLxV8Q)
- 2025-07-03｜**MultiAgentPPT**
  - 說明：多智能體並發產生 PPT
  - 資源：[🐙 GitHub](https://github.com/johnson7788/MultiAgentPPT) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1920611446007497267)
  - 2025-01-13｜**PPTAgent**
  - 說明：中科院開源 AI 工具，文件轉高品質 PPT
  - 資源：[🐙 GitHub](https://github.com/icip-cas/PPTAgent) | [📝 知乎推薦](https://zhuanlan.zhihu.com/p/18105237248)

---

## NotebookLM 平替
**🌍 NotebookLM 平替**

- 2026-01-04 | **Notex**
- 說明：一個開源 NotebookLM 替代方案的實現
  - 資源：[🐙 GitHub](https://github.com/smallnest/notex) | [📝 公眾號推薦](https://mp.weixin.qq.com/s/65epWwIC7Lqalwh-WuoP3Q)
  - [DEMO](https://notex.rpcx.io/)
- 2025-12-06 | **Open NoteBook**
- 說明：一個開源的、注重隱私的Google Notebook LM 替代方案
  - 資源：[🐙 GitHub](https://github.com/smallnest/notex) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1980701578559234518)
- 2025-12-06 | **Auto-Slides**
- 說明：不只是幫你寫，還能幫你講。它讓論文第一次有機會“開口說話”
  - 資源：[🐙 GitHub](https://github.com/Westlake-AGI-Lab/Auto-Slides) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1953393379334391701)

---

## World Models
**🌍 World Models (世界模型)**

- 2025-09-25｜**Code World Model**
  - 說明：Yann LeCun 攜 320 億參數開源世界模型
  - 資源：[📝 Meta Research](https://zread.ai/facebookresearch/cwm/1-overview) | [📝 新浪報導](https://t.cj.sina.com.cn/articles/view/1746173800/68147f6801901e2wa)

---

## MoE
**🧠 MoE (混合專家模型)**

- 2024-12-13｜**DeepSeek-VL2**
  - 說明：VLM 邁入 MoE 時代
  - 資源：[🐙 GitHub](https://github.com/deepseek-ai/DeepSeek-VL2) | [📝 機器之心](https://mp.weixin.qq.com/s/s832KUgixNuX4GUkvY7_Ag) | [📝 公眾號](https://mp.weixin.qq.com/s/p6r_b-k4UnSJED5cBTedZg)

- **騰訊混元 (Hunyuan-Large)**
  - 說明：騰訊最大 MoE 大模型
  - 資源：[🐙 GitHub](https://github.com/Tencent/Hunyuan-Large) | [🤗 DEMO](https://huggingface.co/spaces/tencent/Hunyuan-Large) | [🤗 Model](https://huggingface.co/tencent/Hunyuan-Large) | [📝 機器之心](https://www.jiqizhixin.com/articles/2024-11-06-6)

---

## SLM
**📱 Small Language Models (小型語言模型)**

- 2025-01-07｜**Smolagents**
  - 說明：Hugging Face 全新 AI 智能體框架
  - 資源：[🐙 GitHub](https://github.com/huggingface/smolagents) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/16417392406)

- 2024-12-13｜**Phi-4**
  - 說明：微軟 Phi-4 正式發表，以小博大
  - 資源：[🤗 HuggingFace](https://huggingface.co/NyxKrage/Microsoft_Phi-4) | [📝 公眾號](https://mp.weixin.qq.com/s/uny1VUt7vk_ZU6hCH0EDGg)

- 2024-11-18｜**MobileLLM-1.5B**
  - 說明：Meta 打造行動裝置超強語言模型
  - 資源：[🤗 HuggingFace](https://huggingface.co/facebook/MobileLLM-1.5B) | [📝 公眾號](https://mp.weixin.qq.com/s/hjY6L69pqN4GvybCuHesTA)

- 2024-11-04｜**SmolLM2**
  - 說明：手機執行的小型語言模型
  - 資源：[🤗 HuggingFace](https://github.com/huggingface/smollm/) | [📝 iThome](https://www.ithome.com.tw/news/165832)

- 2024-09-25｜**Llama 3.2**
  - 說明：1B/3B 端側模型 (Edge AI)
  - 資源：[📝 Meta Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

---

## Reasoning
**🤔 Reasoning Models (推理模型)**

- 2025-08-05｜**gpt-oss**
  - 說明：OpenAI 重新開源，o4-mini 水平
  - 資源：[🤗 HuggingFace](https://huggingface.co/openai/gpt-oss-120b) | [📝 OpenAI Blog](https://openai.com/zh-Hant/index/introducing-gpt-oss/) | [📝 機器之心](https://www.jiqizhixin.com/articles/2025-08-06-2)

- 2025-07-29｜**Llama Nemotron Super v1.5**
  - 說明：英偉達開源，三倍吞吐、單卡可跑
  - 資源：[🤗 HuggingFace](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1933514869279274584)

- 2025-07-27｜**OpenReasoning-Nemotron**
  - 說明：英偉達數學核武，1.5B 參數秒殺 o3
  - 資源：[🤗 HuggingFace](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/o7RhRAFzAKkHj2T0y3GVzA)

- 2025-05-06｜**Llama-Nemotron**
  - 說明：英偉達高效推理系列
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.00949) | [📚 DeepWiki](https://deepwiki.com/NVIDIA/NeMo) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1903012593033012833)

- 2025-04-16｜**Video-R1**
  - 說明：Reinforcing Video Reasoning in MLLMs
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2503.21776) | [🐙 GitHub](https://github.com/tulerfeng/Video-R1) | [📝 36Kr 報導](https://www.36kr.com/p/3252742390655489)

---

## LLM
**🏛️ Large Language Models (大型語言模型)**

- 2025-08-05｜**Claude Opus 4.1**
  - 資源：[📝 機器之心](https://www.jiqizhixin.com/articles/2025-08-06-4)

- 2024-11-23｜**Ai2 Tülu 3**
  - 說明：真・開源模型，公開「後訓練」一切
  - 資源：[🐙 GitHub](https://github.com/allenai/open-instruct) | [🌐 Playground](https://playground.allenai.org/) | [🤗 Model](https://huggingface.co/allenai) | [📝 機器之心](https://www.jiqizhixin.com/articles/2024-11-23-5)

- 2024-11-09｜**Ai2 OpenScholar**
  - 資源：[📝 Blog](https://allenai.org/blog/openscholar) | [🌐 Project](https://openscholar.allen.ai/)

- 2024-09-25｜**Llama 3.2 90b/11b**
  - 資源：[📝 Meta Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

---

## Embedding
**🔎 Embedding & Reranker**

- 2025-07-14｜**Gemini Embedding 001**
  - 資源：[☁️ Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings?hl=zh-tw)

- 2025-06-05｜**Qwen3 Embedding**
  - 說明：新一代文本表徵與排序模型
  - 資源：[📝 Qwen Blog](https://qwenlm.github.io/zh/blog/qwen3-embedding/) | [🤗 Embedding](https://huggingface.co/collections/Qwen/qwen3-embedding-6841b2055b99c44d9a4c371f) | [🤗 Reranker](https://huggingface.co/collections/Qwen/qwen3-reranker-6841b22d0192d7ade9cdefea)

---

## Speech
**🔊 Speech-to-Speech LLM (語音大模型)**

- **TEN Agent**
  - 說明：王炸級開源端對端語音模型
  - 資源：[🐙 GitHub](https://github.com/TEN-framework/TEN-Agent) | [📝 公眾號](https://mp.weixin.qq.com/s/pw9LQyRCRogfxAlYG3EfcQ) | [📝 入坑記](https://mp.weixin.qq.com/s/ZVZHNP0XPwzGapWWqTk1kw) | [📝 搭建教學](https://uy6npdpeoi.feishu.cn/docx/EAWYdWWO7ormNPxUhyVcO3GSnUc)

- **pipecat**
  - 說明：用 ChatGPT 即時語音 API 建立應用
  - 資源：[🐙 GitHub](https://github.com/pipecat-ai/pipecat) | [📝 機器之心](https://www.jiqizhixin.com/articles/2025-01-10-4)

- 2025-12-24｜**Fun-Audio-Chat-8B**
  - 資源：[🤗 HuggingFace](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)

- 2025-11-03｜**LongCat-Flash-Omni**
  - 說明：開啟全模態即時互動時代
  - 資源：[🤗 HuggingFace](https://huggingface.co/meituan-longcat/LongCat-Flash-Omni) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1968699530762491165)

- 2025-09-19 | **Xiaomi-MiMo-Audio**
  - 說明：小米開源首個原生端對端語音大模式
  - 資源：[🤗 HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Base) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1991075806194205492)

- 2025-07-21｜**Audio Flamingo 3**
  - 說明：NVIDIA 開源多模態音訊模型
  - 資源：[🐙 GitHub](https://github.com/NVIDIA/audio-flamingo) | [📝 OSChina](https://www.oschina.net/news/361477/nvidia-audio-flamingo-3)

- 2025-05-08｜**Voila**
  - 說明：195ms 超低延遲引領全雙工對話
  - 資源：[🐙 GitHub](https://github.com/maitrix-org/Voila) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1903776373765547954)

- **HuggingFace Speech-to-Speech**
  - 資源：[🐙 GitHub](https://github.com/huggingface/speech-to-speech)

---

## Vision
**👁️ Vision-Language Model (視覺大語言模型)**

- 2025-05-20｜**Seed1.5-VL**
  - 說明：具有視覺增強多模態能力的高階語言模型
  - 資源：[🐙 GitHub](https://github.com/ByteDance-Seed/Seed1.5-VL) | [📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.07062) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1905914968433497765)

- 2025-05-12｜**nanoVLM**
  - 資源：[📚 DeepWiki](https://deepwiki.com/huggingface/nanoVLM)

---

## Multimodal
**🌌 Multimodal LLM (多模態大語言模型)**

- **InternVL**
  - 說明：刷新開源多模態大模型效能新紀錄
  - 資源：[🐙 GitHub](https://github.com/OpenGVLab/InternVL) | [📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2504.10479) | [📚 DeepWiki](https://deepwiki.com/OpenGVLab/InternVL) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1897681159359551408)

- 2025-05-24｜**Dolphin**
  - 說明：開源多模態複雜文件解析模型
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.14059) | [📚 DeepWiki](https://deepwiki.com/bytedance/Dolphin) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/1911355829485045020)

- 2025-05-21｜**Gemma 3n**
  - 資源：[🌐 Google DeepMind](https://deepmind.google/models/gemma/?hl=zh-tw) | [🤗 Preview](https://huggingface.co/google/gemma-3n-E4B-it-litert-preview)

- 2025-03-18｜**Mistral Small 3.1**
  - 說明：128K 上下文，效能碾壓 GPT-4o Mini
  - 資源：[🤗 HuggingFace](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/31138756743)

- 2025-03-14｜**Vision-R1**
  - 說明：激發多模態大模型的推理能力
  - 資源：[🐙 GitHub](https://github.com/Osilly/Vision-R1) | [📝 知乎解讀](https://zhuanlan.zhihu.com/p/29618155786)

- 2025-02-28｜**HumanOmni**
  - 說明：阿里通義業界首個第一視角大模型
  - 資源：[🐙 GitHub](https://github.com/HumanMLLM/HumanOmni) | [📝 公眾號解讀1](https://mp.weixin.qq.com/s/acn16cvE8N4tMegKuGHAKQ) | [📝 公眾號解讀2](https://mp.weixin.qq.com/s/cO6xEAOCRUsLmoiDbq12tw)

- **Phi Family (Microsoft)**
  - 資源：[🤗 Collection](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4) | [🤗 Phi-4 Multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
  - 2025-02-27：[📝 56億參數秒殺 GPT-4o](https://zhuanlan.zhihu.com/p/26984226500) | [📝 小身材大智慧](https://zhuanlan.zhihu.com/p/26678433652)
  - 2024-09-12：[📝 Phi 3.5 mini 發布](https://mp.weixin.qq.com/s/EeALIBrvGWKtEBGnroZIvg)

- **MiniCPM**
  - 資源：[🐙 GitHub](https://github.com/OpenBMB)
  - 2025-01-16：[📝 MiniCPM-o 2.6 發布](https://mp.weixin.qq.com/s/bTRirDr-MCscYF88KmK5qw) | [📖 文檔](https://github.com/OpenBMB/MiniCPM-o/blob/main/README_zh.md#minicpm-o-26)
  - 2024-09-11：[📝 升級 Ollama 支援](https://mp.weixin.qq.com/s/6N-u8PcGEX6e4rryeqXglQ)
  - 2024-09-06：[📝 MiniCPM 3.0 開源](https://53ai.com/news/OpenSourceLLM/2024090659871.html) | [🐙 GitHub](https://github.com/OpenBMB/MiniCPM)
  - 2024-09-05：[📝 魔改 MiniCPM-V](https://mp.weixin.qq.com/s/DjDznmtKZoJNKXYz0X4zog) | [🐙 GitHub](https://github.com/OpenBMB/MiniCPM-V/)


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