---
layout: default
title: 電腦視覺資源懶人包 (Computer Vision) | YOLO, Diffusion & OCR | Deep Learning 101
description: 2025 最新電腦視覺 (CV) 技術資源。包含物件偵測 (YOLO系列)、生成式 AI (Stable Diffusion)、影像分割 (Segmentation) 與 OCR 相關論文與實作教學。
permalink: /Computer-Vision
lang: zh-Hant
schema_type: service
service_type: AI Consulting
---

{% include header.html %}

---

# 👁️ 電腦視覺 (CV)・必讀資源總整理

> **編者按：** 本頁面彙整了電腦視覺領域的關鍵技術資源，涵蓋物件偵測、生成式 AI、影像分割以及文字識別（OCR）等最新論文與實作。
>
> 如果您想尋找更詳細的筆記，歡迎訪問 **GitHub Repository**：
> 👉 [**GitHub: Computer-Vision-Paper**](https://github.com/Deep-Learning-101/Computer-Vision-Paper) (歡迎 Star ⭐)


{% include ai-share.html %}

---

### **文章目錄**
- [Anomaly Detection (異常檢測)](#anomalydetection)
- [Object Detection (目標偵測)](#objectdetection)
- [Segmentation (圖像分割)](#segmentation)
- [OCR (光學文字識別)](#ocr)
- [Diffusion Model (擴散模型)](#diffusion-model)
- [Digital Human (虛擬數字人)](#digital-human)

---

## AnomalyDetection
**Anomaly Detection，異常檢測**

- 2026-01-29｜**LLM2CLIP**
  - 說明：以大語言模型重塑跨模態表徵學習的文本基石
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2411.04997) | [🐙 GitHub](https://microsoft.github.io/LLM2CLIP/) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/-U03e1KZmFCoXTGzdYbC0Q)

- 2025-09-24｜**FS-SAM2**
  - 說明：Adapting Segment Anything Model 2 for Few-Shot Semantic Segmentation
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/overview/2509.12105v1) | [📝 中文解讀：效能與效率雙優](https://zread.ai/fornib/FS-SAM2)

- 2025-09-20｜**MOCHA**
  - 說明：Multi-modal Objects-aware Cross-arcHitecture Alignment
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2509.14001v1) | [📝 中文解讀：注入 YOLO 性能大漲](https://zhuanlan.zhihu.com/p/1952054591035281418)

- 2025-07-16｜**CostFilter-AD**
  - 說明：Enhancing Anomaly Detection through Matching Cost Filtering
  - 資源：[🐙 GitHub](https://github.com/ZHE-SAPI/CostFilter-AD) | [📝 中文解讀：刷新無監督上限](https://zhuanlan.zhihu.com/p/1928870223529882075)

- 2025-06-13｜**One-to-Normal**
  - 說明：Anomaly Personalization (少樣本異常識別新突破)
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2502.01201) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1916799842879018831)

- 2025-06-06｜**DualAnoDiff (CVPR 2025)**
  - 說明：Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2408.13509v3) | [📝 中文解讀：復旦騰訊優圖入選](https://www.qbitai.com/2025/06/291359.html)

- 2025-05-15｜**AdaptCLIP**
  - 說明：Adapting CLIP for Universal Visual Anomaly Detection
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/overview/2407.15795) | [🐙 GitHub](https://github.com/aiiu-lab/AdaptCLIP) | [📝 中文解讀](https://mp.weixin.qq.com/s/w5x6T18aSZt9jxqMIdf-Yg)

- 2025-05-05｜**Multi-Modal LLM for AD**
  - 說明：Detect, Classify, Act: Categorizing Industrial Anomalies
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.02626) | [📚 DeepWiki](https://deepwiki.com/Sassanmtr/VELM) | [💾 Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

- 2025-04-27｜**AnomalyCLIP**
  - 說明：Object-agnostic Prompt Learning for Zero-shot AD
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/overview/2310.18961) | [📚 DeepWiki](https://deepwiki.com/zqhang/AnomalyCLIP)

- 2025-04-26｜**PaDim**
  - 說明：經典無監督異常檢測方法
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2011.08785) | [📚 DeepWiki](https://deepwiki.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)

- 2025-04-12｜**AA-CLIP**
  - 說明：Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2503.06661) | [📚 DeepWiki](https://deepwiki.com/Mwxinnn/AA-CLIP)

- 2025-03-25｜**Dinomaly**
  - 說明：The Less Is More Philosophy in Multi-Class Unsupervised AD
  - 資源：[🐙 GitHub](https://github.com/guojiajeremy/Dinomaly) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1886364053259146390)

---

## ObjectDetection
**Object Detection (目標偵測)**

- 2025｜**MCL (AAAI 2025)**
  - 說明：Multi-clue Consistency Learning (遙感半監督目標檢測)
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2407.05909) | [🐙 GitHub](https://github.com/facias914/sood-mcl) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/26788012528)

- 2025-07-24｜**OV-DINO**
  - 說明：開源工業開放詞彙目標檢測
  - 資源：[🐙 GitHub](https://github.com/wanghao9610/OV-DINO) | [📝 中文解讀](https://mp.weixin.qq.com/s/gLAVYFAH_39gT4XC0zWN0A)

- 2025-06-18｜**CountVid**
  - 說明：Open-World Object Counting in Videos (影片中指哪數哪)
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2506.15368) | [📝 中文解讀](https://mp.weixin.qq.com/s/hICrrfEgriyktoIxnbjPEQ)

- 2025-06-15｜**GeoPix**
  - 說明：像素級遙感多模態大模型
  - 資源：[🐙 GitHub](https://github.com/Norman-Ou/GeoPix) | [📝 北大實驗室介紹](https://3slab.pku.edu.cn/info/1026/2121.htm)

- 2025-05-23｜**VisionReasoner**
  - 說明：用強化學習統一視覺感知與推理 (對標 Qwen2.5-VL)
  - 資源：[🐙 GitHub](https://github.com/dvlab-research/VisionReasoner) | [📝 中文解讀](https://mp.weixin.qq.com/s/vECz3i_-dzvlDr3BdRLPWQ)

- 2025-03-14｜**Falcon**
  - 說明：A Remote Sensing Vision-Language Foundation Model
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2503.11070) | [📚 DeepWiki](https://deepwiki.com/TianHuiLab/Falcon)

---

## Segmentation
**Segmentation (圖像分割)**

- **SAM 3**
  - 資源：[🐙 GitHub](https://github.com/facebookresearch/sam3) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/7uDHXQd1ES2mV4dZFB7VMw)

- **Perceive Anything Model**
  - 說明：Recognize, Explain, Caption, and Segment Anything (對標 SAM2 + LLM)
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2506.05302v1) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1919709726209446971)

- **RemoteSAM**
  - 說明：Towards Segment Anything for Earth Observation
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/abs/2505.18022v3) | [📚 DeepWiki](https://deepwiki.com/1e12Leon/RemoteSAM)

- **InstructSAM**
  - 說明：Training-Free Framework for Remote Sensing
  - 資源：[🌐 Project](https://voyagerxvoyagerx.github.io/InstructSAM/) | [📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.15818v1) | [📚 DeepWiki](https://deepwiki.com/VoyagerXvoyagerx/InstructSAM)

- **SAM 2 & Variants** (SAM 2 相關變體)
  - [**Meta SAM 2**](https://ai.meta.com/sam2/): Meta 官方最新分割一切模型。
    - [📝 60行程式碼微調 SAM 2](https://mp.weixin.qq.com/s/YfgYCzvi0cXxOFIfQvE_9w)
  - [**Grounded SAM 2**](https://github.com/IDEA-Research/Grounded-SAM-2): Ground and Track Anything in Videos.
  - [**SAM2Long**](https://github.com/Mark12Ding/SAM2Long): 港中文提出，專注於複雜長視頻分割。
  - [**SAM2-Adapter**](https://github.com/tianrun-chen/SAM-Adapter-PyTorch): 首次讓 SAM 2 適應下游任務。
  - [**SAM2Point**](https://github.com/ZiyuGuo99/SAM2Point): 可提示 3D 分割研究里程碑。

- **Other Notable Models**
  - [**SAMURAI**](https://yangchris11.github.io/samurai/): KF + SAM2 解決快速移動或自遮擋問題。
  - [**MatAnyone**](https://github.com/pq-yang/MatAnyone): 視訊摳圖，髮絲級還原。
  - [**Exact (CVPR 2025)**](https://github.com/MiSsU-HH/Exact): 遙感影像時間序列弱監督學習。
  - [**SegAnyMo (CVPR 2025)**](https://github.com/nnanhuang/SegAnyMo): Segment Any Motion in Videos.

---

## OCR
**Optical Character Recognition (光學文字識別)**
**[針對物件或場景影像進行分析與偵測](https://www.twman.org/AI/CV)**

- [使用開源模型強化您的 OCR 工作流程](https://huggingface.co/blog/zh/ocr-open-models)
- [12個流行的開源免費OCR項目](https://mp.weixin.qq.com/s/7EuhnQedAX6injBL_Dg_sQ)

- 2026-01-27 | **DeepSeek-OCR 2**
  - 資源：[🐙 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2/) | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | [📝 公眾號解讀](https://mp.weixin.qq.com/s/DOm_hg6DWA_OjcsLuUQ9Hw)

- 2025-11-30｜**HunyuanOCR**
  - 資源：[🐙 GitHub](https://github.com/Tencent-Hunyuan/HunyuanOCR) | [📝 騰訊混元 1B 級全能模型](https://zhuanlan.zhihu.com/p/1977498008712131326)

- 2025-10-21 | **Chandra OCR**
  - 資源：[🐙 GitHub](https://github.com/datalab-to/chandra) | [📝 超越DeepSeek-OCR！ OCR領域的革命性突破：Chandra OCR本地部署+真實評測](https://zhuanlan.zhihu.com/p/1969019468937144099)

- 2025-10-19｜**PaddleOCR-VL**
  - 資源：[🤗 HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | [📝 圖片辨識轉文字巔峰之作](https://zhuanlan.zhihu.com/p/1964600336103745187)

- 2025-08-18｜**DianJin-OCR-R1**
  - 資源：[🐙 GitHub](https://github.com/aliyun/qwen-dianjin) | [📝 點金 OCR-R1：模糊蓋章、跨頁表格全拿下](https://mp.weixin.qq.com/s/cOo0sqwDt3ARid70wBaYVA)

- 2025-07-30｜**dots.ocr**
  - 資源：[🤗 HuggingFace](https://huggingface.co/rednote-hilab/dots.ocr) | [📝 本地部署 1.7B 超強 OCR](https://zhuanlan.zhihu.com/p/1935120171573413613)

- 2025-06-16｜**OCRFlux**
  - 說明：基於 LLM 的複雜佈局與跨頁合併 PDF 解析
  - 資源：[🐙 GitHub](https://github.com/chatdoc-com/OCRFlux) | [🌐 Demo](https://ocrflux.pdfparser.io/#/)

- 2025-06-05｜**MonkeyOCR**
  - 資源：[📚 DeepWiki](https://deepwiki.com/Yuliang-Liu/MonkeyOCR) | [📄 AlphaXiv](https://www.alphaxiv.org/overview/2506.05218)

  - 2025-03-05｜**OpenOCR**
  - 資源：[🐙 GitHub](https://github.com/Topdu/OpenOCR) | [📝 通用OCR工具OpenOCR開源
](https://zhuanlan.zhihu.com/p/10259507246)

- 2025-03-05｜**PP-DocBee**
  - 資源：[🐙 GitHub](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/deploy/ppdocbee) | [📝 百度文檔影像理解](https://zhuanlan.zhihu.com/p/28715553656)

- 2025-03-03｜**olmocr**
  - 資源：[🐙 GitHub](https://github.com/allenai/olmocr) | [📝 本地部署精準提取 PDF](https://www.aivi.fyi/llms/deploy-olmOCR)

- 2025-02-05｜**MinerU**
  - 資源：[🐙 GitHub](https://github.com/opendatalab/MinerU) | [📝 PDF 轉 Markdown 神器](https://mp.weixin.qq.com/s/ci5wp6gICTCtaRZfn5yWUQ)

- 2024-12-15｜**markitdown**
  - 資源：[🐙 GitHub](https://github.com/microsoft/markitdown)

- 2024-10-29｜**OmniParser**
  - 資源：[🐙 GitHub](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/OmniParser) | [📝 Alibaba 出品：通用文檔複雜場景抽取](https://mp.weixin.qq.com/s/_1Aatpna7poIVRhfYk4aAQ)

- 2024-09-11｜**GOT-OCR-2.0**
  - 資源：[📝 模型開源介紹](https://mp.weixin.qq.com/s/rQL-Q0TGhT6e8Ti4zZalrg) | [📝 OCR 2.0 時代來了](https://mp.weixin.qq.com/s/W-Ult-F3pU6Wvx3fHEN8yA)

- 2024-08-20｜**PDF 轉 MarkDown 工具**
  - 資源：[📝 萬物皆可 AI 化！12000 人圍觀的開源工具](https://www.53ai.com/news/MultimodalLargeModel/2024082059736.html)

- **其他實用工具與資源**
  - **RapidOCR**：[🐙 GitHub](https://github.com/RapidAI/RapidOCR/blob/main/docs/README_zh.md)
  - **TableStructureRec**：[🐙 GitHub](https://github.com/RapidAI/TableStructureRec) | [📝 表格結構辨識推理庫](https://zhuanlan.zhihu.com/p/668484933)
  - **PaddleOCR 教學**：[📝 用 PPOCRLabel 微調醫療診斷書和收據](https://blog.twman.org/2023/07/wsl.html)

---


## Diffusion Model
**Diffusion Model (擴散模型)**


- 2025-05-28｜**Jodi**
  - 說明：視覺理解 & 生成大一統模型
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.19084) | [🌐 Project](https://vipl-genun.github.io/Project-Jodi/)

- 2025-05-27｜**AnomalyAny (CVPR 2025)**
  - 說明：Stable Diffusion 協助視覺異常檢測，無需訓練
  - 資源：[🌐 Project](https://hansunhayden.github.io/AnomalyAny.github.io/) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1910284073231942689)

- 2025-05-23｜**HivisionIDPhotos**
  - 說明：智慧證件照生成神器 (摳圖、換背景、任意尺寸)
  - 資源：[📚 DeepWiki](https://deepwiki.com/Zeyi-Lin/HivisionIDPhotos) | [📝 教學文章](https://zhuanlan.zhihu.com/p/718725351)

- 2025-05-19｜**Index-AniSora**
  - 說明：B 站開源 SOTA 動畫影片生成模型
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/overview/2504.10044) | [📚 DeepWiki](https://deepwiki.com/bilibili/Index-anisora) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1908150671540224717)

- 2025-04-26｜**Insert Anything**
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2504.15009) | [📚 DeepWiki](https://deepwiki.com/song-wensong/insert-anything)

- 2025-04-24｜**Phantom**
  - 說明：字節跳動 1280x720 影片生成模型，10G 顯存可用
  - 資源：[🐙 GitHub](https://github.com/Phantom-video/Phantom) | [📝 實測報告](https://zhuanlan.zhihu.com/p/1898688574477545694)

- 2025-04-22｜**MAGI-1**
  - 說明：Sand AI 全球首個自回歸影片生成大模型
  - 資源：[🐙 GitHub](https://github.com/SandAI-org/Magi-1) | [📝 性能亮點解析](https://www.zhihu.com/question/1898030232184795448)

- 2025-04-22｜**SkyReels V2**
  - 說明：全球首個無限時長影片生成，電影級理解
  - 資源：[🐙 GitHub](https://github.com/SkyworkAI/SkyReels-V2) | [📝 媒體報導](https://www.qbitai.com/2025/04/275531.html)

- 2025-04-14｜**FramePack**
  - 說明：ComfyUI 插件，6G 顯存跑 13B 模型，支援 1 分鐘影片
  - 資源：[🐙 GitHub](https://github.com/kijai/ComfyUI-FramePackWrapper) | [📝 性價比分析](https://zhuanlan.zhihu.com/p/1896487969470251546)

- 2025-04-14｜**Fantasy-talking**
  - 說明：基於 Wan2.1 的音訊驅動數字人
  - 資源：[🌐 Project](https://fantasy-amap.github.io/fantasy-talking/) | [📝 解讀文章](https://zhuanlan.zhihu.com/p/1892895916354148118)

- 2025-04-05｜**SkyReels-A2**
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2504.02436) | [📚 DeepWiki](https://deepwiki.com/SkyworkAI/SkyReels-A2) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1892709305301590652)

- 2025-03-10｜**HunyuanVideo-I2V**
  - 說明：騰訊開源圖生視訊模型 + LoRA 訓練腳本
  - 資源：[🐙 GitHub](https://github.com/Tencent/HunyuanVideo-I2V) | [📝 實戰教學](https://zhuanlan.zhihu.com/p/29110060025)

- 2025-02-25｜**Wan-Video**
  - 說明：阿里萬相大模型開源，全模態、全尺寸
  - 資源：[🐙 GitHub](https://github.com/Wan-Video/Wan2.1) | [📝 媒體報導](https://finance.sina.com.cn/jjxw/2025-02-26/doc-inemukxr9127437.shtml)

- 2025-02-14｜**FlashVideo**
  - 說明：字節跳動視訊增強演算法，102 秒生成 1080P 影片
  - 資源：[🐙 GitHub](https://github.com/FoundationVision/FlashVideo) | [📝 解讀文章](https://zhuanlan.zhihu.com/p/23702953115)

- 2025-01-28｜**Sana (ICLR 2025 Oral)**
  - 說明：英偉達/MIT/清華開源，比 FLUX 快 100 倍
  - 資源：[🐙 GitHub](https://github.com/NVlabs/Sana) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/19489214543)

- **Flux & Ecosystem**
  - **Flux Models**: [🤗 Black Forest Labs](https://huggingface.co/black-forest-labs)
    - [Canny-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Canny-dev) | [Depth-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Depth-dev) | [Fill-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Fill-dev) | [Redux-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Redux-dev)
  - **PuLID (2024-11-29)**: [🐙 GitHub](https://github.com/ToTheBeginning/PuLID) | [📝 ComfyUI 教學](https://mp.weixin.qq.com/s/07BMFHaSasl7-PFtkN6_Zg)
  - **Leffa (2024-12-17)**: [🐙 GitHub](https://github.com/franciszzj/Leffa) | [📝 Meta AI 人物特徵保持](https://juejin.cn/post/7449325873725276196)
  - **MagicQuill (2024-11-26)**: [🐙 GitHub](https://github.com/magic-quill/MagicQuill) | [🤗 Space](https://huggingface.co/spaces/AI4Editing/MagicQuill) | [📝 AI P 圖神器](https://mp.weixin.qq.com/s/Pc3xRP8_9BxkVSRNznkplw)

- **Practical Tools (ComfyUI & Others)**
  - **OOTDiffusion**: [🐙 GitHub](https://github.com/levihsu/OOTDiffusion) | [📝 AI 換裝神器](https://mp.weixin.qq.com/s/B2rNCjJLo8coYzoHGPnVaw)
  - **ComfyUI Impact Pack**: [🐙 GitHub](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | [📝 最強臉部修復](https://mp.weixin.qq.com/s/hNQ9BfdGbRQ_Osus-yMJWg)
  - **OmniGen**: [🐙 GitHub](https://github.com/AIFSH/OmniGen-ComfyUI) | [📝 全能影像生成](https://mp.weixin.qq.com/s/msGK0FmNs3T3jbUBHfR9DA)

---

## Digital Human
**Digital Human (虛擬數字人)**

- **Open Avatar Chat**
  - 資源：[📝 專案介紹](https://zread.ai/HumanAIGC-Engineering/OpenAvatarChat) | [📝 GitHub 爆火神器，本地部署無套路](https://mp.weixin.qq.com/s/eNRbU4lZLgdpe_iNSNcfGA)

- **HeyGem**
  - 資源：[🐙 GitHub](https://github.com/GuijiAI/HeyGem.ai) | [📝 數字人克隆神器](https://zhuanlan.zhihu.com/p/29274862393)

- **Duix**
  - 資源：[🐙 GitHub](https://github.com/GuijiAI/duix.ai) | [📝 全球首個真人數字人開源](https://zhuanlan.zhihu.com/p/716583514)

- **Linly-Talker**
  - 說明：結合 LLM 與視覺模型的智能交互系統
  - 資源：[🐙 GitHub](https://github.com/Kedreamix/Linly-Talker)

- **CVPR 2025 / NeurIPS Resources**
  - **EchoMimicV2 (CVPR 2025)**: [🐙 GitHub](https://github.com/antgroup/echomimic_v2) - Striking, Simplified Human Animation.
  - **Hallo3 (CVPR 2025)**: [🐙 GitHub](https://github.com/fudan-generative-vision/hallo3) - Highly Dynamic Portrait Animation.
  - **MimicTalk (NeurIPS 2024)**: [🐙 GitHub](https://github.com/yerfor/MimicTalk) - 3D talking face.

- **Other Tools**
  - **JoyGen**: [🐙 GitHub](https://github.com/JOY-MM/JoyGen) (Audio-Driven 3D Editing)
  - **Latentsync**: [🐙 GitHub](https://github.com/bytedance/LatentSync)
  - **MuseTalk**: [🐙 GitHub](https://github.com/TMElyralab/MuseTalk)

---

## Image Recognition
**Image Recognition (圖像識別)**


- **ViT (Vision Transformer)**
  - 資源：[🐙 GitHub](https://github.com/google-research/vision_transformer) | [📝 解析文章](https://zhuanlan.zhihu.com/p/445122996) | [📝 遷移表現分析](https://zhuanlan.zhihu.com/p/463608959)

- **Swin Transformer**
  - 資源：[🐙 GitHub](https://github.com/microsoft/Swin-Transformer) | [📝 用 CNN 方式打敗 CNN](https://zhuanlan.zhihu.com/p/362690149)

- **EfficientNetV2**
  - 資源：[🐙 GitHub](https://github.com/d-li14/efficientnetv2.pytorch) | [📝 更小更快的訓練](https://zhuanlan.zhihu.com/p/361873583)

---

## Document AI
**Document Understanding & OCR (文檔理解與文字識別)**

- **Donut (2022)**: OCR-free Document Understanding Transformer. [📄 arXiv:2111.15664](./donut.md)
- **LayoutParser (2021)**: Unified toolkit for Deep Learning Based Document Analysis. [📄 arXiv:2103.15348](./LayoutParser.md)
- **TrOCR (2021)**: Transformer-based OCR with Pre-trained Models. [📄 arXiv:2109.10282](./TrOCR.md)
- **DiT (2022)**: Self-supervised Pre-training for Document Image Transformer. [📄 arXiv:2203.02378](./DiT.md)
- **Nougat (2023)**: Neural Optical Understanding for Academic Documents. [📄 arXiv:2308.13418](https://facebookresearch.github.io/nougat/)

<details>
<summary><strong>📚 LayoutLM Series (點擊展開)</strong></summary>

- **LayoutLM (2020)**: Pre-training of Text and Layout. [📄 arXiv:1912.13318](./LayoutLM.md)
- **LayoutLMv2 (2021)**: Multi-modal Pre-training. [📄 arXiv:2012.14740](./LayoutLMv2.md)
- **LayoutXLM (2021)**: Multilingual Visually-rich Document Understanding. [📄 arXiv:2104.08836](./LayoutXLM.md)
- **LayoutLMv3 (2022)**: Pre-training with Unified Text and Image Masking. [📄 arXiv:2204.08387](./LayoutLMv3.md)
</details>

- **Scene Text Recognition**
  - **ABINet (2021)**: Read Like Humans. [📄 arXiv:2103.06495](./ABINet.md)
  - **ABINet++ (2022)**: Iterative Language Modeling for Text Spotting. [📄 arXiv:2211.10578](./ABINet%2B%2B.md)
  - **ABCNet v2 (2021)**: Adaptive Bezier-Curve Network. [📄 arXiv:2105.03620](./ABCNet_v2.md)
  - **SVTR (2022)**: Scene Text Recognition with a Single Visual Model. [📄 arXiv:2205.00159](./SVTR.md)

---

## DeepFake Detection
**DeepFake Detection (深度偽造偵測)**

- **Multi-attentional Deepfake Detection (CVPR 2021)**
  - H. Zhao et al., Proceedings of the IEEE/CVF CVPR 2021.

- **Geometric Features (CVPR 2021)**
  - Improving Efficiency and Robustness through Precise Geometric Features. Sun, Zekun et al.

- **3D Decomposition (CVPR 2021)**
  - Face Forgery Detection by 3D Decomposition. Xiangyu Zhu et al.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://deep-learning-101.github.io/Computer-Vision"
  },
  "headline": "電腦視覺 (Computer Vision) 資源彙整",
  "description": "一份關於電腦視覺（Computer Vision）的詳盡資源清單，內容涵蓋異常檢測、物件偵測、圖像分割、OCR、擴散模型與虛擬數字人等領域的最新研究與開源工具，由台灣深度學習同好會（Deep Learning 101）提供。",
  "image": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg",
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
      "url": "https://raw.githubusercontent.com/Deep-Learning-101/TonTon/refs/heads/main/_includes/DL101-Logo.jpg"
    }
  },
  "datePublished": "2016-11-11",
  "dateModified": "2025-10-21",
  "keywords": "Computer Vision, Anomaly Detection, Object Detection, Segmentation, OCR, Diffusion Model, Digital Human, 電腦視覺, 異常檢測, 物件偵測, 圖像分割, 擴散模型, 虛擬數字人",
  "about": {
    "@type": "Service",
    "serviceType": "AI Consulting",
    "provider": {
      "@type": "Organization",
      "name": "Deep Learning 101, Taiwan"
    },
    "name": "人工智慧顧問服務 (AI Consulting)",
    "description": "提供關於電腦視覺（Computer Vision）領域的專業顧問服務，包含演算法開發、模型選擇、應用落地與技術導入。"
  }
}
</script>