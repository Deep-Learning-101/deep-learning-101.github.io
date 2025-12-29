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

# Computer Vision (CV, 電腦視覺)

{% include ai-share.html %}

---

### **文章目錄**
- [Anomaly Detection](#anomalydetection)
- [Object Detection](#objectdetection)
- [Segmentation](#segmentation)
- [OCR](#ocr)
- [Diffusion model (擴散模型)](#diffusion-model-擴散模型)
- [Digital Human (虛擬數字人)](#digital-human-虛擬數字人)

## AnomalyDetection
**Anomaly Detection，異常檢測**

-2025-09-24：[FS-SAM2: Adapting Segment Anything Model 2 for Few-Shot Semantic Segmentation via Low-Rank Adaptation](https://www.alphaxiv.org/overview/2509.12105v1)；[FS-SAM2靠SAM2+LoRA 實現效能與效率雙優](https://zread.ai/fornib/FS-SAM2)
- 2025-09-20：[MOCHA: Multi-modal Objects-aware Cross-arcHitecture Alignment](https://www.alphaxiv.org/zh/overview/2509.14001v1)；[將大模型多模態語意注入YOLO，少樣本檢測性能大漲](https://zhuanlan.zhihu.com/p/1952054591035281418)
- 2025-07-16：[CostFilter-AD：Enhancing Anomaly Detection through Matching Cost Filtering](https://github.com/ZHE-SAPI/CostFilter-AD)；[刷新無監督異常檢測上限！ CostFilter-AD：首個即插即用的代價濾波for異常檢測範式](https://zhuanlan.zhihu.com/p/1928870223529882075)
- 2025-06-13：[One-to-Normal：Anomaly Personalization](https://www.alphaxiv.org/abs/2502.01201)；[少樣本異常辨識新突破，擴散模型協助精準偵測](https://zhuanlan.zhihu.com/p/1916799842879018831)
- 2025-06-06：[CVPR2025, *DualAnoDiff*：Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](https://www.alphaxiv.org/abs/2408.13509v3)；[以大模型檢測工業品異常，復旦騰訊優圖新演算法入選CVPR 2025](https://www.qbitai.com/2025/06/291359.html)
- 2025-05-15：[**AdaptCLIP**: Adapting CLIP for Universal Visual Anomaly Detection](https://www.alphaxiv.org/overview/2407.15795)；[Github](https://github.com/aiiu-lab/AdaptCLIP)；[騰訊開源AdaptCLIP 模型刷新多領域SOTA](https://mp.weixin.qq.com/s/w5x6T18aSZt9jxqMIdf-Yg)
- 2025-05-05：[Detect, Classify, Act: Categorizing Industrial Anomalies with Multi-Modal Large Language Models](https://www.alphaxiv.org/zh/overview/2505.02626)；[DeepWiki](https://deepwiki.com/Sassanmtr/VELM)；[數據集](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- 2025-04-27：[**AnomalyCLIP**: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection](https://www.alphaxiv.org/overview/2310.18961)；[DeepWiki](https://deepwiki.com/zqhang/AnomalyCLIP)
- 2025-04-26：[PaDim](https://www.alphaxiv.org/zh/overview/2011.08785)；[DeepWiki](https://deepwiki.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
- 2025-04-12：[Anomaly-Aware CLIP, **AA-CLIP**: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP](https://www.alphaxiv.org/zh/overview/2503.06661)；[DeepWiki](https://deepwiki.com/Mwxinnn/AA-CLIP)
- 2025-03-25：[**Dinomaly**：The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection](https://github.com/guojiajeremy/Dinomaly)；[無監督異常檢測（Unsupervised Anomaly Detection，UAD）](https://zhuanlan.zhihu.com/p/1886364053259146390)

## ObjectDetection
**Object Detection (目標偵測)**
- [AAAI2025, Multi-clue Consistency Learning to Bridge Gaps Between General and Oriented Object in Semi-supervised Detection](https://www.alphaxiv.org/abs/2407.05909)；[Github](https://github.com/facias914/sood-mcl)；[AAAI2025 一個遙感半監督目標偵測（半監督旋轉目標偵測）方法](https://zhuanlan.zhihu.com/p/26788012528)
- 2025-07-24：[OV-DINO](https://github.com/wanghao9610/OV-DINO)；[開源工業開放詞彙目標偵測](https://mp.weixin.qq.com/s/gLAVYFAH_39gT4XC0zWN0A)
- 2025-06-18：[CountVid: Open-World Object Counting in Videos](https://www.alphaxiv.org/abs/2506.15368)；[牛津大學開源類別無關的影片目標計數，影片中也能「指哪數哪」](https://mp.weixin.qq.com/s/hICrrfEgriyktoIxnbjPEQ)
- 2025-06-15：[GeoPix](https://github.com/Norman-Ou/GeoPix)；[像素級遙感多模態大模型](https://3slab.pku.edu.cn/info/1026/2121.htm)
- 2025-05-23：[VisionReasoner](https://github.com/dvlab-research/VisionReasoner)；[偵測、分割、計數、問答全拿下？對標Qwen2.5-VL！ VisionReasoner用強化學習統一視覺感知與推理](https://mp.weixin.qq.com/s/vECz3i_-dzvlDr3BdRLPWQ)
- 2025-03-14：[Falcon: A Remote Sensing Vision-Language Foundation Model](https://www.alphaxiv.org/abs/2503.11070)；[DeepWiki](https://deepwiki.com/TianHuiLab/Falcon)
 

## Segmentation
**Segmentation (圖像分割)**
- [Perceive Anything Model：Recognize, Explain, Caption, and Segment Anything in Images and Videos](https://www.alphaxiv.org/zh/overview/2506.05302v1)；[對標SAM2 + LLM融合版！港中文開源感知一切模型與百萬級影像描述資料集：辨識、解釋、描述、分割一體化輸出](https://zhuanlan.zhihu.com/p/1919709726209446971)
- [RemoteSAM](https://www.alphaxiv.org/abs/2505.18022v3)：[Towards Segment Anything for Earth Observation](https://deepwiki.com/1e12Leon/RemoteSAM)
- [InstructSAM](https://voyagerxvoyagerx.github.io/InstructSAM/)：[A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](https://www.alphaxiv.org/zh/overview/2505.15818v1)；[DeepWiki](https://deepwiki.com/VoyagerXvoyagerx/InstructSAM)
- [RESAnything: Attribute Prompting for Arbitrary Referring Segmentation](https://www.alphaxiv.org/abs/2505.02867)；[Project](https://suikei-wang.github.io/RESAnything/)
- [CVPR 2025, Segment Any Motion in Videos, Segment Any Motion in Videos](https://www.alphaxiv.org/zh/overview/2503.22268)；[Github](https://github.com/nnanhuang/SegAnyMo)
- [CVPR 2025 Highlight, Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation](https://www.alphaxiv.org/zh/overview/2412.03968)；[Github](https://github.com/MiSsU-HH/Exact)；[Exact：基於遙感影像時間序列弱監督學習的作物提取方法](https://zhuanlan.zhihu.com/p/38754229963)
- [MatAnyone](https://github.com/pq-yang/MatAnyone)：[視訊摳圖MatAnyone來了，一次指定全程追踪，髮絲級還原](https://www.jiqizhixin.com/articles/2025-04-17-27)
- [Meta Segment Anything Model 2 (SAM 2)](https://ai.meta.com/sam2/)
   - [60行程式碼訓練/微調Segment Anything 2](https://mp.weixin.qq.com/s/YfgYCzvi0cXxOFIfQvE_9w)
   - [CLIPSeg：Image Segmentation Using Text and Image Prompts](https://github.com/timojl/clipseg)：[Huggingface Space](https://huggingface.co/spaces/taesiri/CLIPSeg)
      - [哥廷根大學提出CLIPSeg，能同時作三個分割任務的模型](https://mp.weixin.qq.com/s/evKssKulZiUssLN71t6_Lw)
      - [SAM與CLIP強強聯手，實現22000類的分割與識別](https://mp.weixin.qq.com/s/evKssKulZiUssLN71t6_Lw)
- [SAMURAI](https://yangchris11.github.io/samurai/)
   - [無需訓練或微調即可得到穩定、準確的追蹤效果！ KF + SAM2 解決快速移動或自遮擋的物件追蹤問題](https://mp.weixin.qq.com/s/iU3Bk_uO01GWUxAtIBsrWQ)
   - [經典卡爾曼濾波器改進影片版「分割一切」，網友：好優雅的方法](https://www.qbitai.com/2024/11/223020.html)
- [Grounded SAM 2: Ground and Track Anything in Videos](https://github.com/IDEA-Research/Grounded-SAM-2)
   - [Grounded-Segment-Anything](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything)
- [SAM2Long](https://github.com/Mark12Ding/SAM2Long)：[大幅提升SAM 2性能！港中文提出SAM2Long，複雜長視頻的分割模型](https://mp.weixin.qq.com/s/henvaxGoNgx24NLQV1Qj2w)
- [SAM2-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)：[SAM 2無法分割一切？ SAM2-Adapter：首次讓SAM 2在下游任務適應調校！](https://mp.weixin.qq.com/s/3z-LshKAgbSzNCzyoLOuag)
- [SAM2Point](https://github.com/ZiyuGuo99/SAM2Point)：[可提示3D 分割研究里程碑！ SAM2Point：SAM2加持可泛化任3D場景、任意提示！](https://mp.weixin.qq.com/s/TnTK5UE7O_hcrNzloxBmAw)

- [Optical Character Recognition，光學文字識別](#OCR)

## OCR
**Optical Character Recognition (光學文字識別)**  
**[針對物件或場景影像進行分析與偵測](https://www.twman.org/AI/CV)**

- 2025-11-30：[HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)；[OCR新晉頂流！騰訊混元開源1B級全能模式HunyuanOCR，效果驚艷
](https://zhuanlan.zhihu.com/p/1977498008712131326)
- 2025-10-21：[使用開源模型強化您的 OCR 工作流程](https://huggingface.co/blog/zh/ocr-open-models)
- 2025-10-19：[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)；[PaddleOCR-VL可太強了，圖片辨識轉文字的巔峰之作](https://zhuanlan.zhihu.com/p/1964600336103745187)
- 2025-08-18：[DianJin-OCR-R1](https://github.com/aliyun/qwen-dianjin)；[點金OCR-R1，模糊蓋章、跨頁表格、文字幻覺全拿下！](https://mp.weixin.qq.com/s/cOo0sqwDt3ARid70wBaYVA)
- 2025-07-30：[dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)；[本地部署1.7B参数超强OCR大模型dots.ocr](https://zhuanlan.zhihu.com/p/1935120171573413613)
- 2025-06-16：[OCRFlux](https://github.com/chatdoc-com/OCRFlux)；[DEMO](https://ocrflux.pdfparser.io/#/)；[OCRFlux：一個基於LLM的複雜佈局與跨頁合併的PDF文檔解析](https://mp.weixin.qq.com/s/UgkKLYsVxrrFy-5Gn0JD-g)
- 2025-06-05：[MonkeyOCR](https://deepwiki.com/Yuliang-Liu/MonkeyOCR)；[Document Parsing with a Structure-Recognition-Relation Triplet Paradigm](https://www.alphaxiv.org/overview/2506.05218)
- 2025-05-21：[PaddleOCR 3.0](https://github.com/PaddlePaddle/PaddleOCR/tree/release/3.0)；[OCR精準度躍升13%，支援多語種、手寫體與高精準度文件解析](https://zhuanlan.zhihu.com/p/1908447391784342470)
- 2025-03-05：[PP-DocBee](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/deploy/ppdocbee)：[百度推出文件影像理解PP-DocBee](https://zhuanlan.zhihu.com/p/28715553656)
- 2025-03-03：[olmocr](https://github.com/allenai/olmocr)：[🚀本地部署最强OCR大模型olmOCR！支持结构化精准提取复杂PDF文件内容！](https://www.aivi.fyi/llms/deploy-olmOCR)
- 2025-02-05：[MinerU](https://github.com/opendatalab/MinerU)：[將PDF轉換為機器可讀格式的神器](https://mp.weixin.qq.com/s/ci5wp6gICTCtaRZfn5yWUQ)
- 2024-12-15：[markitdown](https://github.com/microsoft/markitdown)
- 2024-09-22：[OCR2.0时代-GOT来啦！](https://mp.weixin.qq.com/s/W-Ult-F3pU6Wvx3fHEN8yA)
- 2024-09-11：[GOT-OCR-2.0模型开源](https://mp.weixin.qq.com/s/rQL-Q0TGhT6e8Ti4zZalrg)
- 2024-08-20：[萬物皆可AI化！剛開源就有12000人圍觀的OCR 掃描PDF 開源工具！還可轉換為MarkDown！](https://www.53ai.com/news/MultimodalLargeModel/2024082059736.html)
- [advancedliteratemachinery/OCR/OmniParser](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/OmniParser)
- 2024-10-29：[Alibaba出品:OmniParser通用文檔複雜場景下OCR抽取](https://mp.weixin.qq.com/s/_1Aatpna7poIVRhfYk4aAQ)
- [RapidOCR](https://github.com/RapidAI/RapidOCR/blob/main/docs/README_zh.md)
- [12個流行的開源免費OCR項目](https://mp.weixin.qq.com/s/7EuhnQedAX6injBL_Dg_sQ)
- [用PaddleOCR的PPOCRLabel來微調醫療診斷書和收據](https://blog.twman.org/2023/07/wsl.html)
- [TableStructureRec: 表格結構辨識推理庫來了](https://zhuanlan.zhihu.com/p/668484933)：https://github.com/RapidAI/TableStructureRec

## Diffusion model (擴散模型)
- 2025-05-28：[視覺理解&生成大一統模型 Jodi](https://vipl-genun.github.io/Project-Jodi/)；[alphaXiv](https://www.alphaxiv.org/zh/overview/2505.19084)
- 2025-05-27：[AnomalyAny](https://hansunhayden.github.io/AnomalyAny.github.io/)；[CVPR2025｜突破資料瓶頸！ Stable Diffusion 協助視覺異常檢測，無需訓練即可產生真實多樣異常樣本](https://zhuanlan.zhihu.com/p/1910284073231942689)
- 2025-05-23：[HivisionIDPhotos，智慧證件照產生神器](https://deepwiki.com/Zeyi-Lin/HivisionIDPhotos)；[AI證件照，摳圖、換背景、任意尺寸](https://zhuanlan.zhihu.com/p/718725351)
- 2025-05-19：[Index-AniSora](https://deepwiki.com/bilibili/Index-anisora)；[Aligning Anime Video Generation with Human Feedback](https://www.alphaxiv.org/overview/2504.10044)；[B站開源SOTA動畫影片生成模型Index-AniSora！](https://zhuanlan.zhihu.com/p/1908150671540224717)
- 2025-04-26：[Insert Anything](https://www.alphaxiv.org/zh/overview/2504.15009)；[DeepWiki](https://deepwiki.com/song-wensong/insert-anything)  
- 2025-04-24：[字節Phantom](https://github.com/Phantom-video/Phantom)：[1280x720影片生成革命！位元組Phantom模型實測：10G顯存效果不輸某靈付費版](https://zhuanlan.zhihu.com/p/1898688574477545694)
- 2025-04-22：[MAGI-1](https://github.com/SandAI-org/Magi-1)：[Sand AI 創業團隊推出了全球首個自回歸影片生成大模型MAGI-1，該模型有哪些效能亮點？](https://www.zhihu.com/question/1898030232184795448)
- 2025-04-22：[SkyReels V2](https://github.com/SkyworkAI/SkyReels-V2)：[全球首個無限時長影片生成！新擴散模式引爆兆市場，電影級理解，全面開源](https://www.qbitai.com/2025/04/275531.html)
- 2025-04-14：[FramePack](https://github.com/kijai/ComfyUI-FramePackWrapper)：[不是可靈用不起，而是FramePack更有性價比！開源專案：6G顯存跑13B模型，支援1分鐘影片產生](https://zhuanlan.zhihu.com/p/1896487969470251546)
- 2025-04-14：[fantasy-talking](https://fantasy-amap.github.io/fantasy-talking/)：[解讀最新基於Wan2.1的音訊驅動數位人FantasyTalking](https://zhuanlan.zhihu.com/p/1892895916354148118)
- 2025-04-05：[SkyReels-A2](https://www.alphaxiv.org/zh/overview/2504.02436)；[DeepWiki](https://deepwiki.com/SkyworkAI/SkyReels-A2)；[SkyReels-A2：用AI重新定义视频创作的未来！](https://zhuanlan.zhihu.com/p/1892709305301590652)
- 2025-03-10：[HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V)：[騰訊開源HunyuanVideo-I2V圖生視訊模型+LoRA訓練腳本，社群部署、推理實戰教學來吧](https://zhuanlan.zhihu.com/p/29110060025)
- 2025-02-25：[Wan-Video](https://github.com/Wan-Video/Wan2.1)：[超越Sora！阿里萬相大模型正式開源！全模態、全尺寸大模型開源](https://finance.sina.com.cn/jjxw/2025-02-26/doc-inemukxr9127437.shtml)
- 2025-02-14：[FlashVideo](https://github.com/FoundationVision/FlashVideo)：[來自位元組的視訊增強全新開源演算法，102秒產生1080P視頻](https://zhuanlan.zhihu.com/p/23702953115)
- 2025-01-28：[Sana](https://github.com/NVlabs/Sana)：[ICLR 2025 Oral] Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer；[比FLUX快100倍！英偉達聯手MIT、清華開源超快AI影像產生模型](https://zhuanlan.zhihu.com/p/19489214543)
- [Flux](https://huggingface.co/black-forest-labs)
   - [Flux.1-canny-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-canny-dev)：[https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/)
   - [Flux.1-depth-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Depth-dev)：[https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/)
   - [Flux.1-fill-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Fill-dev)：[https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/)
   - [Flux.1-redux-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Redux-dev)：[https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/)
      - 2024-11-26：[Flux官方重繪+擴圖+風格參考+ControlNet](https://mp.weixin.qq.com/s/Kj1nyJNTpoZ94JjO4FMw_g)
      - 2024-11-25：[最新flux_fill_inpaint模型體驗。](https://mp.weixin.qq.com/s/OPknDJXH1_oezSR86c_png)
- 2024-12-17：[Leffa](https://github.com/franciszzj/Leffa)：[Leffa：Meta AI 開源精確控制人物外觀和姿勢的圖像生成框架，在生成穿著的同時保持人物特徵](https://juejin.cn/post/7449325873725276196)
- 2024-11-29：[PuLID, Pure and Lightning ID Customization via Contrastive Alignment](https://github.com/ToTheBeginning/PuLID)：[https://github.com/balazik/ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
   - 2024-11-07：[搞定ComfyUI-PuLID-Flux節點只要這幾步！附一鍵壓縮包](https://mp.weixin.qq.com/s/07BMFHaSasl7-PFtkN6_Zg)
   - 2024-10-08：[一文搞懂PuLID FLUX人物換臉&風格遷移](https://mp.weixin.qq.com/s/V-2Cp8_xFnHQNFn35aGdLg)
- 2024-11-26：[MagicQuill](https://github.com/magic-quill/MagicQuill)：[https://huggingface.co/spaces/AI4Editing/MagicQuill](https://huggingface.co/spaces/AI4Editing/MagicQuill)
   - [MagicQuill，登上Huggingface趨勢榜榜首的AI P圖神器](https://mp.weixin.qq.com/s/Pc3xRP8_9BxkVSRNznkplw)
- 2024-11-26：[OOTDiffusion](https://github.com/levihsu/OOTDiffusion)：[https://huggingface.co/spaces/levihsu/OOTDiffusion](https://huggingface.co/spaces/levihsu/OOTDiffusion)
   - [開源AI換裝神器OOTDiffusion](https://mp.weixin.qq.com/s/B2rNCjJLo8coYzoHGPnVaw)
- 2024-11-24：[Comfyui Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
   - [Comfyui 最強臉部修復工具Impact Pack](https://mp.weixin.qq.com/s/hNQ9BfdGbRQ_Osus-yMJWg)
- 2024-11-05：[ComfyUI OmniGen @ 北京人工智慧研究院](https://github.com/AIFSH/OmniGen-ComfyUI)：[https://huggingface.co/spaces/Shitao/OmniGen](https://huggingface.co/spaces/Shitao/OmniGen)
   - [ComfyUI 影像生成模型OmniGen，人物一致性處理的也太好了](https://mp.weixin.qq.com/s/msGK0FmNs3T3jbUBHfR9DA)
   - [全能影像生成模型OmniGen：告別ControlNet、ipadapter等插件，僅憑提示即可控制影像生成與編輯](https://mp.weixin.qq.com/s/48HmqRGBOK1uBdzlprdKSA)


## Digital Human (虛擬數字人)
- [Open Avatar Chat](https://zread.ai/HumanAIGC-Engineering/OpenAvatarChat)；[GitHub 爆火數位人神器！讓你擁有專屬AI 分身，語音對話+ 即時表情+ 本地部署，免費開源無套路](https://mp.weixin.qq.com/s/eNRbU4lZLgdpe_iNSNcfGA)
- [HeyGem](https://github.com/GuijiAI/HeyGem.ai)：[開源數位人克隆神器](https://zhuanlan.zhihu.com/p/29274862393)
- [Duix](https://github.com/GuijiAI/duix.ai)：[全球首個真人數位人，開源了](https://zhuanlan.zhihu.com/p/716583514)
- [Linly-Talker](https://github.com/Kedreamix/Linly-Talker)：an intelligent AI system that combines large language models (LLMs) with visual models to create a novel human-AI interaction method. 
- [EchoMimicV2](https://github.com/antgroup/echomimic_v2)：[CVPR 2025] EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation
- [Hallo3](https://github.com/fudan-generative-vision/hallo3)：[CVPR 2025] Highly Dynamic and Realistic Portrait Image Animation with Diffusion Transformer Networks
- [MimicTalk](https://github.com/yerfor/MimicTalk)：[NeurIPS 2024] MimicTalk: Mimicking a personalized and expressive 3D talking face in minutes
- [JoyGen](https://github.com/JOY-MM/JoyGen)：Audio-Driven 3D Depth-Aware Talking-Face Video Editing
- [Latentsync](https://github.com/bytedance/LatentSync)
- [MuseTalk](https://github.com/TMElyralab/MuseTalk)

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