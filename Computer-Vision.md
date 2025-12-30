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
- [Diffusion Model (擴散模型)](#diffusion-model-擴散模型)
- [Digital Human (虛擬數字人)](#digital-human-虛擬數字人)

---

## AnomalyDetection
**Anomaly Detection，異常檢測**

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

- 2025-11-30｜**HunyuanOCR**
  - 資源：[🐙 GitHub](https://github.com/Tencent-Hunyuan/HunyuanOCR) | [📝 騰訊混元 1B 級全能模型](https://zhuanlan.zhihu.com/p/1977498008712131326)

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

- 2025-03-05｜**PP-DocBee**
  - 資源：[🐙 GitHub](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/deploy/ppdocbee) | [📝 百度文檔影像理解](https://zhuanlan.zhihu.com/p/28715553656)

- 2025-03-03｜**olmocr**
  - 資源：[🐙 GitHub](https://github.com/allenai/olmocr) | [📝 本地部署精準提取 PDF](https://www.aivi.fyi/llms/deploy-olmOCR)

- 2025-02-05｜**MinerU**
  - 資源：[🐙 GitHub](https://github.com/opendatalab/MinerU) | [📝 PDF 轉 Markdown 神器](https://mp.weixin.qq.com/s/ci5wp6gICTCtaRZfn5yWUQ)

---

## Diffusion model (擴散模型)

- 2025-05-28｜**Jodi**
  - 說明：視覺理解 & 生成大一統模型
  - 資源：[🌐 Project](https://vipl-genun.github.io/Project-Jodi/) | [📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2505.19084)

- 2025-05-27｜**AnomalyAny**
  - 說明：(CVPR 2025) Stable Diffusion 協助視覺異常檢測
  - 資源：[🌐 Project](https://hansunhayden.github.io/AnomalyAny.github.io/) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/1910284073231942689)

- 2025-05-23｜**HivisionIDPhotos**
  - 說明：智慧證件照產生神器 (摳圖、換背景)
  - 資源：[📚 DeepWiki](https://deepwiki.com/Zeyi-Lin/HivisionIDPhotos) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/718725351)

- 2025-05-19｜**Index-AniSora**
  - 說明：B站開源 SOTA 動畫影片生成模型
  - 資源：[📚 DeepWiki](https://deepwiki.com/bilibili/Index-anisora) | [📄 AlphaXiv](https://www.alphaxiv.org/overview/2504.10044)

- 2025-04-26｜**Insert Anything**
  - 資源：[📄 AlphaXiv](https://www.alphaxiv.org/zh/overview/2504.15009) | [📚 DeepWiki](https://deepwiki.com/song-wensong/insert-anything)

- 2025-04-22｜**SkyReels V2**
  - 說明：全球首個無限時長影片生成
  - 資源：[🐙 GitHub](https://github.com/SkyworkAI/SkyReels-V2) | [📝 媒體報導](https://www.qbitai.com/2025/04/275531.html)

- 2025-03-10｜**HunyuanVideo-I2V**
  - 說明：騰訊開源圖生視訊模型 + LoRA 訓練
  - 資源：[🐙 GitHub](https://github.com/Tencent/HunyuanVideo-I2V) | [📝 部署實戰](https://zhuanlan.zhihu.com/p/29110060025)

- 2025-02-25｜**Wan-Video (萬相)**
  - 說明：阿里開源全模態、全尺寸影片生成模型
  - 資源：[🐙 GitHub](https://github.com/Wan-Video/Wan2.1) | [📝 媒體報導](https://finance.sina.com.cn/jjxw/2025-02-26/doc-inemukxr9127437.shtml)

- 2025-01-28｜**Sana (ICLR 2025)**
  - 說明：比 FLUX 快 100 倍的生成模型 (NVlabs)
  - 資源：[🐙 GitHub](https://github.com/NVlabs/Sana) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/19489214543)

- **Flux Family (Black Forest Labs)**
  - [Flux.1-canny-dev](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/) | [Depth](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/) | [Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/)
  - [📝 Flux 官方重繪+擴圖+ControlNet 解讀](https://mp.weixin.qq.com/s/Kj1nyJNTpoZ94JjO4FMw_g)

---

## Digital Human (虛擬數字人)

- **Open Avatar Chat**
  - 說明：語音對話 + 即時表情 + 本地部署 AI 分身
  - 資源：[📝 介紹文章](https://zread.ai/HumanAIGC-Engineering/OpenAvatarChat) | [📝 媒體報導](https://mp.weixin.qq.com/s/eNRbU4lZLgdpe_iNSNcfGA)

- **HeyGem**
  - 說明：開源數位人克隆神器
  - 資源：[🐙 GitHub](https://github.com/GuijiAI/HeyGem.ai) | [📝 中文解讀](https://zhuanlan.zhihu.com/p/29274862393)

- **Linly-Talker**
  - 說明：LLM + Visual Models 智慧互動系統
  - 資源：[🐙 GitHub](https://github.com/Kedreamix/Linly-Talker)

- **EchoMimicV2 (CVPR 2025)**
  - 說明：Towards Striking, Simplified, and Semi-Body Human Animation
  - 資源：[🐙 GitHub](https://github.com/antgroup/echomimic_v2)

- **Hallo3 (CVPR 2025)**
  - 說明：Highly Dynamic and Realistic Portrait Image Animation
  - 資源：[🐙 GitHub](https://github.com/fudan-generative-vision/hallo3)

- **Latentsync** & **MuseTalk**
  - 資源：[🐙 Latentsync](https://github.com/bytedance/LatentSync) | [🐙 MuseTalk](https://github.com/TMElyralab/MuseTalk)

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