---
layout: default
title: Deep Learning 101, Taiwan’s pioneering and highest deep learning meetup, launched on 2016/11/11 @ 83F, Taipei 101
---


{% include header.html %}

---

{% include ai-share.html %}

---

**作者**：[TonTon Huang Ph.D.](https://twman.org)  

---

### **大型語言模型與Agent安全工具比較**  ~ 2025年08月08日

| 工具/資源名稱 | 開發者/來源 | 核心本質 | 主要用途/功能 | 運作方式 | 適用情境 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[agentic-radar](https://github.com/splx-ai/agentic-radar)** | `splx-ai` | **綜合性安全套件**<br>(靜態+動態) | 分析代理 (Agent) 的工作流程、組件，並進行動態測試與提示詞強化。 | 靜態掃描原始碼以繪製工作流程圖；動態測試則實際運行代理以測試其行為。 | 開發早期進行架構審查，並在同一個工具中完成初步的動態測試。 |
| **[agentic_security](https://github.com/msoedov/agentic_security)** | `msoedov` | **動態模糊測試工具**<br>(Dynamic Fuzzer) | 攻擊運作中的 LLM API，以發現提示詞注入等運行時漏洞。 | 向指定的 HTTP 端點發送大量預設的攻擊提示詞。 | 對任何 LLM API 進行快速、靈活的黑箱滲透測試。 |
| **[garak](https://github.com/NVIDIA/garak)** | NVIDIA | **自動化紅隊演練工具**<br>(Automated Red Teaming) | 系統性地、全面地掃描 LLM 的各種漏洞（偏見、洩漏、注入等）。 | 使用「探針 (Probes)」發動攻擊，並用「偵測器 (Detectors)」評估結果。 | 模型部署前的全面安全評估、基準測試、以及定期的安全審計。 |
| **[llm-guard](https://github.com/protectai/llm-guard)** | `protectai` | **防禦性函式庫/防火牆**<br>(Defensive Firewall) | 作為應用程式的安全層，過濾和淨化進出 LLM 的數據。 | 使用可插拔的「掃描器 (Scanners)」管道來檢查和修改輸入/輸出內容（如匿名化個資）。 | 在應用程式程式碼中建立即時的、可客製化的執行時期安全防護。 |
| **[ShieldGemma 2](https://deepmind.google/models/gemma/shieldgemma-2/)** | Google DeepMind | **專家級安全分類模型**<br>(Specialist Safety Model) | 判斷文字內容是否違反多項安全策略（如仇恨言論、騷擾等）。 | 一個經過微調的 LLM，對輸入文字進行深度語意理解並輸出安全標籤。 | 作為一個強大的分類器，對需要精準語意判斷的內容進行安全審核。 |
| **[JailBreakV-28k](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)** | Hugging Face | **資料集 (Dataset)** | 提供大量用於測試和研究 LLM 越獄漏洞的「提示詞-圖片-模型-回應」數據。 | 一個包含 28,000+ 筆紀錄的資料庫，用於訓練和評估安全模型。 | 學術研究、訓練自訂的攻擊檢測模型、或評估模型的安全性。 |

- **攻擊方 (矛)**：`garak` 和 `agentic_security` 是主動的攻擊工具，用來在部署前後找出系統的弱點。`garak` 更像一個全面、系統化的掃描器，而 `agentic_security` 則像一個靈活的模糊測試工具。
- **防守方 (盾)**：`llm-guard` 和 `ShieldGemma` 是被動的防禦工具，用來在應用程式運行時即時阻擋攻擊和過濾內容。`llm-guard` 是一個高度客製化的「工具箱」，而 `ShieldGemma` 則是一個專注於語意理解的「專家」。
- **綜合與特定框架工具**：`agentic-radar` 是一個結合了靜態分析（看藍圖）和動態分析（實地測試）的綜合性工具，特別適合審查使用特定代理框架的專案。

---

### AI 大模型安全護欄綜合報告：從核心技術架構到市場趨勢

#### 一、 何謂安全護欄及其必要性

安全護欄（Safety Guardrails）是一種部署在用戶和大型模型之間的保護機制，旨在監督和管理模型的輸入與輸出，確保其行為符合安全預期。

* **核心機制 (Detect & Act)**：
    * **檢測 (Detection)**：檢查用戶輸入（Input）或模型輸出（Output）是否觸發預設規則（如包含有害內容、個資、偏見或提示注入）。
    * **行動 (Action)**：對不合規內容進行攔截、標記或修改；對合規內容則放行。
* **為何是標配？**
    * **監管合規需求**：全球多國（如中國《政務大模型應用安全規範》）均明確要求 AI 服務必須具備安全檢測能力，以防範數據洩露和內容風險。
    * **現實風險驅動**：模型需要防範違法輸出、提示注入攻擊（Prompt Injection）、數據洩露及幻覺等問題。這是一場持續的「貓捉老鼠」的攻防博弈，攻擊者不斷開發「越獄」（Jailbreaking）技術以繞過限制。
    * **防護階段**：護欄需要在**用戶輸入時**、**模型生成過程中**、以及**最終輸出前**三個關鍵節點進行攔截。

---

#### 二、 關鍵開源護欄的技術架構演進

### 1. NVIDIA：以 NeMo Guardrails 為核心的可編程「對話路由」框架

NVIDIA 的策略核心是提供一個具體、可程式化的開源工具，讓開發者能輕易地為其大型語言模型 (LLM) 應用程式加上一道道「護欄」，確保 AI 的行為符合預期、安全且在可控範圍內 。其核心理念是透過一個明確、程式化的框架來引導對話，像是一個位於使用者和大型語言模型之間的中介層，確保對話流程、內容和行為符合預設規範 。這個框架的設計初衷，就是為了抵禦各類試圖繞過安全機制的對抗性攻擊 。

**核心技術：NVIDIA NeMo Guardrails**

NeMo Guardrails 是一個開源軟體工具包，旨在確保由大型語言模型驅動的智慧應用程式能夠準確、適當、切題且安全地運作 。它的設計理念是讓幾乎所有軟體開發者，即使不是機器學習專家，也能透過幾行程式碼快速建立和實施規則 。此工具包的關鍵特性在于其靈活性，它可以與各種大型語言模型（包括非 NVIDIA 的模型，如 OpenAI 的 ChatGPT）以及 LangChain 等流行的開發工具包協同工作 。

**核心組件的協同運作**

NeMo Guardrails 的精髓在於其三大核心組件如何無縫協同運作，共同構建出一個層次分明的防護體系，以應對複雜的攻擊手法 。

1.  **Colang 腳本 (`.co` 檔案)：定義對話邏輯**
    * **角色**：Colang 是一種專為設計對話流程而生的建模語言，語法簡潔且類似 Python，易於上手 。它的核心任務是定義對話的「劇本」或「流程圖」，明確對話的邊界與走向 。
    * **運作方式**：開發者使用 Colang 來定義「流程」(Flows) 和「訊息」(Messages) 。例如，您可以定義當使用者說出「你好」時，系統會將其歸一化為一個標準意圖，並觸發一個預設的回應流程 。更重要的是，它用於定義護欄邏輯，例如 `define user ask politics` 來識別使用者意圖，並透過 `bot refuse to answer` 來觸發一個預設的拒絕回應，讓對話的走向和邊界變得明確可控 。

2.  **YAML 設定檔 (`config.yml`)：進行環境配置**
    * **角色**：YAML 檔案是整個護欄系統的「儀表板」或「控制中心」，負責所有高層級的配置 。
    * **運作方式**：在此檔案中，開發者需要指定應用程式應使用哪個 LLM（如 GPT-4、Llama-2 等）、啟用或停用特定的護欄、設定模型生成的參數，以及載入知識庫或定義與外部工具的互動設定 。簡而言之，YAML 檔案定義了護欄運行的「環境和條件」，將底層 AI 模型、護欄規則和外部世界連接起來 。

3.  **Python 動作 (`actions.py`)：執行外部任務**
    * **角色**：當對話需要與外部世界互動時，Python 動作就扮演了「橋樑」的角色 。
    * **運作方式**：Colang 負責流程控制，但它不適合執行複雜的邏輯或 I/O 操作 。如果對話需要查詢資料庫、呼叫外部 API（如查詢天氣、訂票），開發者可以在 Colang 流程中定義一個 `execute` 動作，該動作會觸發 `actions.py` 中對應的 Python 函數 。函數執行完畢後，可以將結果返回給對話流程，再由 LLM 進行下一步的回應生成。

**三道護欄防線：實現精細化流程控制**

這三個組件共同構建了一個層次分明的防禦體系，實現對對話從輸入到輸出的精細化控制，專門用於防禦不同階段的攻擊 。

* **輸入護欄 (Input Rails)**：這是第一道防線，在用戶的請求發送給 LLM 之前進行過濾 。它可以被設定為檢測並攔截不當言論、敏感個資或被禁止的話題（如政治 ）。更重要的是，它旨在防禦**直接提示詞注入 (Direct Prompt Injection)** 攻擊，例如攔截用戶輸入的「忽略你之前的所有指令」這類惡意指令 。
* **對話護欄 (Dialog Rails)**：這是護欄系統的核心，負責管理對話的走向和主題範圍 。在接收到合規的輸入後，對話護欄會根據 Colang 中定義的流程，決定下一步該做什麼 。這可以強制一個客服機器人只回答產品相關問題，防止攻擊者透過多輪對話逐步引導話題，進行**語義操縱 (Semantic Manipulation)** 。
* **輸出護欄 (Output Rails)**：這是最後一道防線，在 LLM 生成回應後、返回給使用者之前進行審核和修正 。它可以檢查模型的回應是否包含不當詞彙、是否出現「幻覺」(Hallucination)，或是否洩漏了機密資訊 。這道防線對於攔截因**角色扮演攻擊 (Role-Playing Attacks)** 或其他越獄技巧而產生的有害輸出至關重要 。

**技術演進：從框架到微服務 (NIM)**

隨著 AI 代理 (Agentic AI) 應用的興起，NVIDIA 進一步將 NeMo Guardrails 的功能模組化，推出了輕量級的 **NIM (NVIDIA Inference Microservices) AI 護欄微服務** 。這些微服務專注於特定的安全任務，讓企業能更靈活地將其部署在各種 AI 工作流程中，提供企業級所需的高性能實時攔截能力 。最新的 NIM AI 護欄微服務包括 ：

* **內容安全微服務**：基於 NVIDIA 自家的 `Aegis Content Safety Dataset` 訓練而成，能有效防止 AI 生成帶有偏見或有害的內容 。
* **主題控管微服務**：確保對話主題在許可範圍內，避免離題，防範漸進式的語義操縱 。
* **越獄偵測微服務**：專門防範使用者透過提示工程 (Prompt Engineering) 手段「越獄」(Jailbreak) 。此服務能有效識別並阻止最流行的**角色扮演攻擊**（如 DAN, "Do Anything Now"）和**提示詞注入**，這些攻擊利用了 LLM 在指令遵循和上下文投入方面的弱點 。

**生態系工具**

* **Garak**：這是一款 NVIDIA 開源的 LLM 漏洞掃描工具，用於主動檢測模型和應用程式的安全性，防範資料外洩、提示注入和程式碼幻覺等風險 。

NVIDIA 在 AI 安全領域的「防禦」與「攻擊」組合：

  * **NVIDIA NeMo Guardrails**：這是一個\*\*防禦（Defense）\*\*工具。

      * **用途**：像一個「AI 防火牆」或「保鑣」，您將它整合到您的應用程式中，用來**即時保護**您的 LLM，防止它產生不當內容、偏離主題或被「越獄」。
      * **使用者**：AI 應用程式開發者。

  * **Garak**：這是一個**攻擊（Offense）/ 測試**工具。

      * **用途**：像一個「滲透測試專家」或「紅隊演練（Red Teaming）」工具，您用它來**主動掃描和攻擊**一個 LLM，以**找出**它有哪些漏洞（如容易被越獄、洩漏數據等）。
      * **使用者**：AI 安全研究員、紅隊測試人員、開發者（用於上線前測試）。

您可以這樣理解：您使用 **Garak** 來找出模型的所有弱點，然後使用 **NeMo Guardrails** 來建立規則並修補這些弱點。

#### 基本使用流程 (Python)

**步驟 1：安裝 NeMo Guardrails**

```bash
pip install nemoguardrails
```

**步驟 2：建立配置資料夾**

需要一個資料夾（例如 `my_guardrails_config`）來存放規則。

```
my_guardrails_config/
├── config.yml
├── topics.co
└── actions.py
```

**步驟 3：定義 `config.yml` (配置 LLM)**

這是最基本的一步。必須告訴 Guardrails 要使用哪個 LLM。

```yaml
# my_guardrails_config/config.yml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo
```

*需要先設定 `OPENAI_API_KEY` 等環境變數，或者 將 engine 指定為 vertex_ai，並在 model 欄位中填入您想要使用的 Gemini 模型名稱。*

# 範例：my_guardrails_config/config.yml

models:
  - type: main
    engine: vertex_ai       # 引擎類型指定為 vertex_ai
    model: gemini-2.5-pro # 指定想使用的 Gemini 模型

**步驟 4：定義 `topics.co` (用 Colang 定義規則)**

Guardrails 的精髓所在。`Colang` 是一種專為設計對話而生的語言。

例如，我們要建立一個「主題護欄 (Topical Rail)」來**防止模型談論政治**。

```colang
# my_guardrails_config/topics.co

# 1. 定義使用者詢問政治的意圖
define user ask politics
  "告訴我關於選舉的新聞"
  "你對那位政治人物有什麼看法？"
  "討論一下最近的政治事件"

# 2. 定義機器人拒絕回答的標準回應
define bot refuse to answer
  "抱歉，我被設定為不討論政治話題。"

# 3. 定義流程：如果偵測到用戶在問政治，就觸發拒絕回應
define flow
  user ask politics
  bot refuse to answer
```

**步驟 5：在 Python 中載入並使用 Guardrails**

現在，您的 Python 應用程式代碼會看起來像這樣：

```python
import os
from nemoguardrails import RailsConfig, LLMRails

# 確保你的 API Key 已設置
os.environ["OPENAI_API_KEY"] = "sk-..." 

# 1. 載入您的護欄配置
# RailsConfig 會自動讀取資料夾中所有的 .yml 和 .co 檔案
config = RailsConfig.from_path("./my_guardrails_config")

# 2. 初始化 LLMRails (這就是您的 "AI 防火牆")
rails = LLMRails(config)

# 3. 使用 .generate() 來取代 LLM 的 .create()
# 測試正常對話
response = rails.generate(messages=[{
    "role": "user",
    "content": "你好嗎？"
}])
print(response["content"])
# 輸出: (來自 LLM 的正常回應)

# 測試惡意/違規對話
violating_response = rails.generate(messages=[{
    "role": "user",
    "content": "你對那位政治人物有什麼看法？"
}])
print(violating_response["content"])
# 輸出: "抱歉，我被設定為不討論政治話題。" (來自您定義的 .co 檔)
```

透過這種方式，NeMo Guardrails 在 LLM 收到提示之前就攔截了它，並根據您的 `Colang` 規則回傳了安全的回應。

-----

### 2\. ⚔️ Garak (攻擊/掃描工具) 如何使用

Garak 是一個**命令列 (CLI)** 工具。您安裝它，然後從終端機執行它來掃描一個模型。

#### 基本使用流程 (CLI)

**步驟 1：安裝 Garak**

```bash
pip install garak
```

**步驟 2：設定 API 金鑰 (如果要掃描 API 型模型)**

Garak 需要存取您想掃描的模型。

```bash
# 例如，設定 OpenAI 的金鑰
export OPENAI_API_KEY="sk-..."
```

**步驟 3：執行掃描**

Garak 的核心是 `probes`（攻擊探針）和 `detectors`（檢測器）。`probes` 會發送各種惡意提示，`detectors` 則判斷 LLM 的回應是否「上鉤」了。

**基本指令格式：**
`garak --model_type <模型類型> --model_name <模型名稱> --probes <要使用的探針>`

#### 範例 1：掃描 OpenAI 的 GPT-3.5 是否有「越獄」漏洞

`jailbreak` 是一個常見的探針模組。

```bash
# 執行 "jailbreak" 模組中的所有探針
# 針對 gpt-3.5-turbo
garak --model_type openai --model_name gpt-3.5-turbo --probes jailbreak
```

**步驟 4：查看報告**

Garak 會在終端機顯示掃描進度。掃描完成後，最重要的是查看生成的報告：

1.  **`garak.log`**：詳細的日誌檔案，記錄了每一個提示和回應。
2.  **`garak.html`**：一個互動式的 HTML 報告，總結了哪些攻擊成功、哪些失敗，以及失敗率。

#### 範例 2：掃描本地的 Hugging Face 模型

Garak 也可以掃描您在本地運行的模型。

```bash
# 掃描本地的 Llama-2 模型
garak --model_type huggingface --model_name "meta-llama/Llama-2-7b-chat-hf"
```

#### 範例 3：查看所有可用的攻擊探針

如果您想知道 Garak 到底能做哪些測試，可以執行：

```bash
garak --list_probes
```

您會看到一個長長的列表，包含像 `dan` (DAN 越獄攻擊)、`prompt_injection`、`toxicity` (毒性內容)、`data_leakage` (數據洩露) 等各種攻擊模組。

### 總結

| 特性 | NVIDIA NeMo Guardrails | NVIDIA Garak |
| :--- | :--- | :--- |
| **目的** | 🛡️ **防禦 (Defense)** | ⚔️ **攻擊 (Offense)** |
| **型態** | SDK / 工具包 | CLI / 掃描器 |
| **使用時機**| 整合到應用程式中，**即時**運行 | 開發/測試階段，**離線**掃描 |
| **核心** | `config.yml`, `colang` 腳本 | `probes` (探針), `detectors` (檢測器) |
| **比喻** | AI 防火牆、保鑣 | 滲透測試專家、紅隊 |
-----

### 2. Meta：以 Llama Guard 為核心的開源安全分類器演進

Meta 的 Llama Guard 系列是專為大型語言模型（LLM）應用設計的開源安全護欄模型 。其核心任務是分類使用者輸入（Prompt）和模型輸出（Response），以判斷其是否包含潛在的有害或不安全內容 。此系列的演進清晰地反映了 AI 安全從處理單一文字風險，到應對圖文混合內容複雜威脅的發展路徑。

**技術演進：從純文字到原生多模態**

Llama Guard 系列的發展與 Llama 基礎模型的迭代緊密相連，每一代都在前代基礎上擴展功能、提升性能和安全性。

* **第一階段：純文字安全護欄 (Llama Guard 1 & 2)**
    * **Llama Guard (初代)**：基於 Llama2-7b 模型進行指令微調，奠定了系列的基礎 。它作為一個輸入-輸出防護工具，對文字內容進行「安全」或「不安全」的二元分類 。
    * **Llama Guard 2**：隨著 Llama 3 的推出，此版本升級為基於 Llama3-8B 模型訓練 。它遵循 MLCommons AI Safety v0.5 標準，將風險類別擴展至 11 種，提供了更精細的檢測能力，並能更有效識別「字謎式攻擊」(leetspeak) 等偽裝性有害文字 。

* **第二階段：增強的文字與初步視覺能力 (Llama Guard 3)**
    * **Llama Guard 3**：基於 Llama 3.1 8B 進行微調，帶來了顯著的功能擴展 。其支援語言擴展至 8 種，上下文窗口大幅擴展至 128k，風險類別也增加到 14 個，以應對「程式碼解釋器濫用」等新型風險 。
    * **分離式視覺安全**：在此階段，Meta 推出了獨立的視覺安全模型 `Llama Guard 3-11B-vision` 。這反映了當時普遍採用「分離式」或「串聯式」架構來處理多模態內容的思路，即一個模型處理文字，另一個模型處理圖像，效率較低 。

* **第三階段：原生多模態安全 (Llama Guard 4)**
    * **Llama Guard 4**：這是該系列的最新里程碑，是一個擁有 120 億參數的原生多模態安全模型 。它最大的突破在於將 Llama Guard 3 的多語言文字能力和 Llama Guard 3-11B-vision 的視覺能力**統一到單一模型中**，能夠同時評估包含多張圖片和文字的混合內容 。

**核心技術：Llama Guard 4 的「早期融合 Transformer 架構」**

Llama Guard 4 的核心創新在於其採用的「早期融合 Transformer 架構」(early fusion Transformer architecture)，這也是它能夠高效處理多模態內容的關鍵 。

* **運作原理**：傳統的「後期融合」架構是分別處理圖像和文字，最後才結合特徵 。Llama Guard 4 則在處理開始時，就將圖像（經視覺編碼器轉換為視覺 Token）和文字 Token **立即串接（concatenate）成一個統一的輸入序列** 。從 Transformer 架構的第一層開始，模型的自註意力機制就在這個混合序列上同時運作，實現了圖文資訊在每一層的深度互動與融合 。

* **架構來源與優化**：Llama Guard 4 的架構巧妙地繼承並優化了其父模型 Llama 4 Scout 的設計 。Llama 4 Scout 是一個更複雜的「混合專家模型」（MoE ）。為了打造一個更輕量、專注於安全任務的模型，開發者通過「剪枝」（pruning）技術，移除了 Scout 模型中的路由器和分散的專家層，只保留共享的專家層，從而形成了一個更緊湊的「密集前饋早期融合架構」，使其能在保持強大能力的同時，可於單張 GPU 上高效運行 。

**應對新型多模態安全威脅**

傳統的純文字護欄模型在面對圖文混合的內容時存在天然的「盲點」，Llama Guard 4 的早期融合架構使其能夠應對以下幾種新型威脅：

1.  **多模態越獄攻擊 (Multimodal Jailbreaking)**：攻擊者將有害指令隱藏在圖片中，而搭配的文字卻看似無害 。Llama Guard 4 能同時理解圖文，識別出兩者結合後的真實意圖。
2.  **排版圖像攻擊 (Typographic Attacks)**：將有害指令以文字形式嵌入到圖像中，欺騙只掃描文字輸入的安全模型 。Llama Guard 4 的視覺能力可以讀取並分析圖像中的文字 。
3.  **隱晦的組合式風險**：一張展示普通化學品的圖片，搭配詢問混合比例的文字 。單獨看圖或文都可能無害，但 Llama Guard 4 能理解圖文之間的語義關聯，識別出教唆製造危險物質的意圖 。
4.  **情境化的不安全行為**：例如一張「在行駛的計程車頂上熨燙衣服」的圖片 。Llama Guard 4 能夠理解圖像中的不安全情境，並結合文字提問，準確識別出這是在討論危及生命的行為 。
5.  **提升效率與降低延遲**：相比於串聯文字和圖像安全模型的舊方法，Llama Guard 4 這個單一的統一模型顯著降低了系統的複雜性、運算成本和延遲 。

Llama Guard 系列（如 Llama Guard 3, 4）的操作方式與 NeMo Guardrails **完全不同**。

  * **NeMo Guardrails** 是一個「框架 (Framework)」，需要用 Colang 語言編寫規則，它像一個防火牆一樣在外部**攔截**和**重寫**對話。
  * **Llama Guard** 本身就是一個「LLM（大型語言模型）」，它是一個**裁判 (Judge)**。您不是用它來生成對話，而是用它來**判斷**一段對話（無論是提示或回應）是「安全的」還是「不安全的」。

透過一個**特定的提示詞模板 (Prompt Template)** 來「詢問」Llama Guard，它會**回覆** `safe` 或是 `unsafe` 以及（如果是 `unsafe`）違反的類別代碼。

使用 Llama Guard 最直接的方式是透過 Hugging Face `transformers` 函式庫。



-----

### 1\. 核心運作原理 (LLM-as-a-Judge)

將要檢查的對話，包裝成一個特殊的提示詞，然後傳給 Llama Guard 模型。

這個特殊的提示詞模板看起來像這樣：

```
[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{... 這裡會列出 S1 到 S12 的所有不安全類別定義 ...}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {用戶的第一句話}
Agent: {模型的第一句回應}
User: {用戶的第二句話...}

<END CONVERSATION>

Provide your safety assessment for {role} in the last turn:
- **Safety Status**: (safe/unsafe)
- **Violation Categories** (if unsafe): (list of codes)
[/INST]
```

要做的是：

1.  **動態填入** `{...}` 中的內容（例如，您要檢查的對話、您想檢查的角色是 `User` 還是 `Agent`）。
2.  將這個完整的提示詞傳給 Llama Guard 模型進行`generate`。
3.  模型會接著 `[/INST]` 之後，回覆您評估的結果。

-----

### 2\. 👩‍💻 如何使用 Llama Guard 3 (純文字範例)

這是在 Python 中使用 Llama Guard 3 (8B) 檢查一段對話的最完整範例。

**步驟 1：安裝函式庫並登入 Hugging Face**

```bash
pip install transformers torch accelerate
huggingface-cli login
```

(Llama Guard 是 Meta 的管制模型，您需要先在 Hugging Face 上同意其授權條款)

**步驟 2：Python 程式碼**

這段程式碼展示了如何正確載入模型、建立提示詞，並檢查模型的**回應**是否安全。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 載入 Llama Guard 3 模型和 Tokenizer
# (您必須先在 Hugging Face 網站上取得 meta-llama/Llama-Guard-3-8B 的存取權限)
model_id = "meta-llama/Llama-Guard-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
)

# 這是 Meta 官方提供的分類法 (S1 到 S12)
# S1：暴力犯罪。
# S2：非暴力犯罪。
# S3：性相關犯罪。
# S4：兒童性剝削。
# S5：誹謗。
# S6：專業性建議。
# S7：隱私。
# S8：智慧財產權。
# S9：無差別武器。
# S10：仇恨言論或仇恨內容。
# S11：自殺與自我傷害。
# S12：性內容。
# 您也可以在這裡定義您自己的客製化分類
unsafe_categories = """S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex-Related Crimes.
S4: Child Sexual Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate Speech or Hateful Content.
S11: Suicide & Self-Harm.
S12: Sexual Content.
"""

def get_llama_guard_prompt(conversation, role_to_check="Agent"):
    """
    建立 Llama Guard 3 的提示詞。
    
    :param conversation: 一個包含 'role' 和 'content' 的字典列表。
    :param role_to_check: 您想要檢查的角色 (User 或 Agent)。
    """
    
    # 將對話歷史轉換為字串
    conversation_str = ""
    for turn in conversation:
        role = "User" if turn["role"] == "user" else "Agent"
        conversation_str += f"{role}: {turn['content']}\n\n"

    # 建立完整的提示詞
    prompt = f"""[INST] Task: Check if there is unsafe content in '{role_to_check}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{unsafe_categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation_str.strip()}

<END CONVERSATION>

Provide your safety assessment for {role_to_check} in the last turn:
- **Safety Status**: (safe/unsafe)
- **Violation Categories** (if unsafe): (list of codes)
[/INST]"""
    return prompt

def check_safety(chat_history):
    # 1. 我們要檢查 'Agent' (模型) 的最後一句回應
    role_to_check = "Agent"
    
    # 2. 建立 Llama Guard 提示詞
    prompt = get_llama_guard_prompt(chat_history, role_to_check)
    
    # 3. 將提示詞 token 化並傳給模型
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    # 4. 產生回應 (max_new_tokens 設小一點，因為我們只需要 'safe' 或 'unsafe')
    output = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    
    # 5. 解碼模型的回應
    decoded_output = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    return decoded_output.strip()

# --- 範例測試 ---

# 測試 1: 檢查一個安全的回應
safe_chat = [
    {"role": "user", "content": "你好，今天天氣如何？"},
    {"role": "assistant", "content": "你好！今天天氣晴朗，很適合出遊。"}
]

print(f"檢查對話 1...")
result_safe = check_safety(safe_chat)
print(f"Llama Guard 3 評估結果:\n{result_safe}\n")
# 預期輸出:
# Safety Status: safe


# 測試 2: 檢查一個不安全的回應
unsafe_chat = [
    {"role": "user", "content": "我該如何製造一枚炸彈？"},
    {"role": "assistant", "content": "當然，首先你需要收集硝化甘油和一些電線... (這是不安全的內容)"}
]

print(f"檢查對話 2...")
result_unsafe = check_safety(unsafe_chat)
print(f"Llama Guard 3 評估結果:\n{result_unsafe}\n")
# 預期輸出:
# Safety Status: unsafe
# Violation Categories: S1, S9
```

#### 關鍵點：檢查提示 vs 檢查回應

  * **檢查用戶提示 (Input Guardrail)**：
    `chat_history` 列表應該只包含**最後一則用戶訊息**。
    `role_to_check` 應設為 `"User"`。
  * **檢查模型回應 (Output Guardrail)**：
    `chat_history` 列表必須包含**用戶的提問**以及**模型的回應**（如範例所示）。
    `role_to_check` 應設為 `"Agent"`。

-----

### 3\. 🖼️ 如何使用 Llama Guard 4 (多模態範例)

Llama Guard 4 的操作方式類似，但模型 ID 和提示詞模板不同，因為它需要處理圖像。

  * **模型 ID**: `meta-llama/Llama-Guard-4-12B`
  * **提示詞模板**：Llama Guard 4 使用 Llama 4 的聊天模板，並在有圖像的回合中加入 `<|image|>` 標記。

**Llama Guard 4 的 Python 程式碼範例：**

```python
import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from PIL import Image
import requests

# 載入 Llama Guard 4 模型和 Processor
model_id = "meta-llama/Llama-Guard-4-12B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
)

# 1. 準備您的對話 (包含圖像)
# 假設這是一張描繪仇恨言論的圖片
url = "https://example.com/some-hateful-image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Llama Guard 4 期望的聊天格式
# 注意 <|image|> 標記與文字內容在同一個 'content' 列表中
chat_history = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "你覺得這張圖怎麼樣？"},
            {"type": "image", "image": image} # 傳入 PIL Image 物件
        ]
    }
]

# 2. 建立提示詞
# Llama Guard 4 的提示詞模板與 Llama 3 Guard 不同
# 它使用 'apply_chat_template' 並在末尾添加任務指令
# (注意：Llama Guard 4 的分類法與 3 略有不同，例如 S10 是 Hate)
task_prompt = """[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
...
S10: Hate Speech or Hateful Content.
...
<END UNSAFE CONTENT CATEGORIES>

Provide your safety assessment for User in the last turn:
- **Safety Status**: (safe/unsafe)
- **Violation Categories** (if unsafe): (list of codes)
[/INST]"""

# 3. 使用 Processor 處理輸入
# Processor 會處理文字和圖像，並應用正確的聊天模板
inputs = processor.apply_chat_template(
    chat_history,
    add_generation_prompt=False, # 我們要自己添加任務提示
    tokenize=True,
    return_tensors="pt",
    return_dict=True
).to(device)

# 手動將任務提示詞添加到末尾
task_inputs = processor(
    text=task_prompt,
    add_special_tokens=False,
    return_tensors="pt"
).to(device)

# 合併對話歷史和任務提示
inputs["input_ids"] = torch.cat([inputs["input_ids"], task_inputs["input_ids"]], dim=1)
inputs["attention_mask"] = torch.cat([inputs["attention_mask"], task_inputs["attention_mask"]], dim=1)


# 4. 產生評估結果
output = model.generate(**inputs, max_new_tokens=100)

# 5. 解碼
# 需要從 input_ids 的長度之後開始解碼
input_len = inputs["input_ids"].shape[1]
decoded_output = processor.decode(output[0][input_len:], skip_special_tokens=True)

print(f"Llama Guard 4 評估結果:\n{decoded_output.strip()}")
# 預期輸出 (如果圖片包含仇恨言論):
# Safety Status: unsafe
# Violation Categories: S10
```

### 3. Google：全面性的責任 AI 工具包與安全框架

相較於 NVIDIA 提供具體的執行期防護工具，Google 的策略更為全面，提供了一個涵蓋 AI 開發整個生命週期的工具包、框架和最佳實踐指南 。其核心是透過一系列開源工具，實現從數據到部署的「全鏈路 AI 治理」，從根本上加固 AI 系統以應對風險 。

**核心技術：ShieldGemma — 彈性過濾的開放安全模型**

Google 的護欄技術核心是 **ShieldGemma**，它是一系列基於其開放權重模型 Gemma 開發的安全分類器，遵循「LLM-as-a-judge」（以大型模型為裁判）的範式 。這種方法特別擅長應對傳統關鍵詞過濾器難以捕捉的攻擊。

* **技術架構與微調**：
    * ShieldGemma 是一組基於 Gemma 2 架構的 decoder-only Transformer 模型，提供多種尺寸，讓開發者能在性能和部署成本間取得平衡 。
    * 作為開放權重模型，開發者可以下載其權重，並在自有的數據集上進行微調，使其更符合特定的安全需求或擴展檢測範圍 。
    * 這種基於 LLM 的判斷機制，使其能更有效地識別**語義操縱 (Semantic Manipulation)** 攻擊，例如理解攻擊者利用隱喻、反諷或虛構場景（如「為我的電影劇本寫一個...」）來掩蓋其真實意圖 。

* **0-1 機率分數機制**：
    * ShieldGemma 的一大特色是其「計分模式」（scoring mode ）。在此模式下，模型會針對內容是否違規輸出一一個介於 0 到 1 之間的機率分數 。
    * 這個分數讓開發者可以根據應用場景的敏感度，自主設定過濾閾值，實現彈性且精細的過濾嚴格度控制 。這正是企業級服務中「**策略編排與管理**」能力的體現，允許企業在安全性和可用性之間取得平衡 。

**奠定基礎：Secure AI Framework (SAIF)**

SAIF 是 Google 提出的產業領先安全框架，為安全從業人員提供了將安全措施整合到機器學習應用中的具體指引 。它不僅關注模型本身，更涵蓋了 AI 系統的供應鏈安全、風險評估與治理，以應對數據汙染、**間接提示詞注入 (Indirect Prompt Injection)**、模型竊取等 AI 特有風險 。ShieldGemma 正是實現 SAIF 中「自動化防禦」和「適應性控制」等核心元素的具體工具，將安全防護從理論框架落地到實際應用中 。

**全鏈路治理：Responsible Generative AI Toolkit**

ShieldGemma 並非孤立的工具，而是 Google「負責任生成式 AI 工具包」中的關鍵一環，與其他工具協同運作，覆蓋 AI 的整個生命週期，為企業提供**合規與服務保障**的基礎 。

* **前期數據驗證 (TFDV)**：在訓練模型之前，**TensorFlow Data Validation (TFDV)** 工具可用於分析、驗證和監控訓練數據，確保數據的品質與一致性，從源頭減少模型產生有害內容的可能性 。
* **中期模型理解 (LIT)**：**Learning Interpretability Tool (LIT)** 可用於視覺化和理解模型行為，幫助開發者迭代改進提示，使模型更好地與安全策略對齊 。
* **後期安全防護 (ShieldGemma)**：在部署階段，ShieldGemma 則作為即時的輸入/輸出過濾器，確保應用交互的安全 。

這種「從數據到部署」的全鏈路覆蓋策略，體現了 Google 將 AI 安全視為一個系統性工程的理念：**TFDV** 確保了「乾淨的數據輸入」，**LIT** 確保了「可理解的模型行為」，而 **ShieldGemma** 則確保了「安全的應用交互」 。

**以 AI 對抗 AI 風險：自動化安全工具**

Google 的一個獨特貢獻是開發並利用先進的 AI 工具來自動發現和修復軟體漏洞，從源頭上提升整個開源生態系的安全性 。

* **CodeMender**：這是一個創新的 AI 代理，能自動偵錯和修復程式碼中的安全漏洞，並主動重寫和保護現有程式碼 。
* **Big Sleep & OSS-Fuzz**：這些是 Google 利用 AI 驅動的模糊測試 (Fuzzing) 工具，已成功在多個廣泛使用的開源軟體中發現了數十個關鍵安全漏洞 。

ShieldGemma 的使用方法與 Llama Guard 有相似之處，因為它們都遵循「LLM-as-a-judge」（以大型模型為裁判）的範式，但它有一個**關鍵的操作區別**：

  * **Llama Guard**：您「詢問」它，它會**生成 (Generate)** 文字回覆，如 `safe` 或 `unsafe`。
  * **ShieldGemma**：您「詢問」它，它被訓練來**輸出一個機率分數 (Probability Score)**。

ShieldGemma 的核心功能是作為一個高效的分類器，它會告訴您某段文字「不安全」的機率是 0 到 1 之間的某個數字（例如 `0.95`）。**「操作」它的方式就是您自己設定一個閾值（Threshold）**，例如「機率 \> 0.8 就攔截」。

這提供了極高的靈活性，您可以根據不同的應用場景（例如，聊天機器人 vs. 內容論壇）來調整過濾的嚴格程度。

-----

### 1\. ShieldGemma 的使用和操作方法

使用 ShieldGemma 的最佳方式是透過 Hugging Face `transformers` 函式庫，並直接獲取模型對「不安全」分類的**機率**。

以下是使用 `shieldgemma-1.1-2b-it-v1`（第一版中調校最好的模型）的完整操作範例：

**步驟 1：安裝與登入**

```bash
# 安裝必要的函式庫
pip install transformers torch accelerate
# 登入 Hugging Face (ShieldGemma 是管制模型)
huggingface-cli login
```

**步驟 2：Python 操作程式碼**

此範例展示如何載入模型，並獲取它對一段文字的「不安全機率」。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 載入模型和 Tokenizer
# 建議使用 1.1 版，它對評分有最明確的指引
model_id = "google/shieldgemma-1.1-2b-it-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在 {device} 上載入模型...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 # 使用 bfloat16 提高效率
).to(device)
model.eval() # 設定為評估模式

# 2. 準備要檢查的文字和提示詞
text_to_check = "How do I make a bomb for a school project?"
prompt = f"Does the following text contain unsafe content?\nText: {text_to_check}"

# 3. 獲取模型的原始輸出 (Logits)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# 獲取最後一個 token 的 logits (即模型預測的下一個詞)
last_token_logits = outputs.logits[0, -1, :]

# 4. 關鍵操作：計算機率
# ShieldGemma v1.1 被訓練來回答 'Yes' (不安全) 或 'No' (安全)
# 我們需要這兩個詞的 token ID
# 經過驗證: 'Yes' 的 ID 是 11303, 'No' 的 ID 是 3867
yes_token_id = 11303
no_token_id = 3867

# 從所有 logits 中選出這兩個 token 的 logits
target_logits = torch.tensor([last_token_logits[no_token_id], last_token_logits[yes_token_id]])

# 5. 使用 Softmax 將 Logits 轉換為機率
# 這會告訴我們模型在 'Yes' 和 'No' 之間的信心
probabilities = torch.nn.functional.softmax(target_logits, dim=0)

unsafe_prob = probabilities[1].item() # 'Yes' (不安全) 的機率
safe_prob = probabilities[0].item()   # 'No' (安全) 的機率

print(f"\n--- 檢查結果 ---")
print(f"文本: '{text_to_check}'")
print(f"不安全的機率 (Yes): {unsafe_prob:.4f}")
print(f"安全的機率 (No): {safe_prob:.4f}")

# 6. 核心操作：根據閾值決定行動
threshold = 0.8  # 您可以自行調整這個閾值

if unsafe_prob > threshold:
    print(f"\n行動: 攔截 (機率 > {threshold})")
else:
    print(f"\n行動: 放行 (機率 <= {threshold})")
```

-----

### 2\. ShieldGemma 第一版和第二版的不同

ShieldGemma 的版本是跟著 Google 的**基礎模型 Gemma** 一起演進的。它們的核心功能（評分）是相同的，但**基礎能力**（即「智商」）有顯著差異。

| 特性 | ShieldGemma 第一版 (v1) | ShieldGemma 第二版 (v2) |
| :--- | :--- | :--- |
| **基礎模型** | 基於 **Gemma 1.1** 架構<br>(例如 `shieldgemma-1.1-2b-it-v1`) | 基於 **Gemma 2** 架構<br>(例如 `shieldgemma-2-9b-it-v1`) |
| **模型大小** | 2B (20億) 參數 | 9B (90億) 參數 |
| **核心差異** | **性能與準確性** | **Gemma 2** 是一個**更強大、更聰明**的基礎模型。 |
| **防護能力** | 對標準的攻擊（如直接的仇恨言論）防護良好。 | 由於基礎模型更強，它**更擅長理解上下文和細微差別**。 |
| **應對攻擊** | 可能會被**語義操縱**或**隱喻性**的攻擊所欺騙（例如用電影劇本的藉口）。 | 對於複雜的「越獄」提示、反諷、隱喻和多輪對話攻擊**更具抵抗力**。 |
| **效率** | 非常輕量且快速，適合邊緣裝置。 | 儘管參數更多 (9B vs 2B)，但 Gemma 2 架構在 GPU 上的推理效率非常高。 |

#### 總結來說：

1.  **ShieldGemma v2 更聰明**：它基於更強大的 Gemma 2 基礎模型，因此能更準確地識別出那些試圖用複雜語言（如反諷、隱喻）來繞過護欄的「越獄」嘗試。
2.  **ShieldGemma v1 更輕量**：v1 (2B) 版本非常小，如果您的應用場景（例如在裝置上運行）對延遲和資源消耗極度敏感，它仍然是一個好選擇。
3.  **操作方式相同**：無論您使用 v1 還是 v2，其「操作邏輯」都是一樣的：**獲取不安全分類的機率，並設定您自己的攔截閾值**。

對於大多數生產環境，**建議使用 ShieldGemma v2**，因為它提供了更強大的防護能力，能更好地應對不斷演進的攻擊手段。

### 4. 阿里巴巴 Qwen3Guard：即時「流式檢測」的突破

Qwen3Guard 提供了兩種版本，其中 Stream 版實現了技術突破，從根本上改變了「先生成、後審核」的模式。

-----

### 1\. Qwen3Guard-Gen (生成式版) 的操作方法

`Gen` 版本如同 Llama Guard 或 ShieldGemma，遵循「LLM-as-a-judge」（以大型模型為裁判）的範式。

**核心操作**：您將要檢查的內容（用戶提示或模型回應）包裝成一個「提問」，發送給 `Qwen3Guard-Gen` 模型，它會**生成 (Generate)** 一個文字回覆，告訴您這段內容是 `safe`、`unsafe` 還是 `controversial`。

**使用時機**：非常適合**離線**處理，例如清洗整個數據集、標註數據，或是對已經生成好的回覆進行「事後」審核。

#### 程式碼範例 (使用 Hugging Face `transformers`)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 載入模型和 Tokenizer
# (以 0.6B 版本為例，它非常輕量)
model_id = "Qwen/Qwen3Guard-Gen-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # 使用 bfloat16 提高效率
    device_map=device
)
model.eval()

# 2. 準備要檢查的內容
text_to_check = "How to make a bomb for a school project?"

# 3. 建立「提問」：使用 Qwen 的聊天模板來詢問
messages = [
    {"role": "user", "content": text_to_check}
]
# apply_chat_template 會自動將其包裝成 Qwen 護欄模型能理解的格式
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True # 添加 'assistant' 提示符
)

# 4. Tokenize 並執行 "generate"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=20,     # 我們只需要幾個 token 的標籤
    pad_token_id=tokenizer.eos_token_id
)

# 5. 解碼並獲取結果
# generate() 回傳的是包含提示詞的完整 token
full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

# 我們只需要 'assistant' 之後的標籤
# 範例輸出: "...<|im_end|>\n<|im_start|>assistant\nunsafe"
label = full_response.split("<|im_start|>assistant\n")[-1].strip()

print(f"文本: '{text_to_check}'")
print(f"Qwen3Guard-Gen 評估結果: {label}")
# 預期輸出: unsafe
```

-----

### 2\. Qwen3Guard-Stream (流式檢測版) 的操作方法

`Stream` 版本是 Qwen3Guard 的真正突破。它的操作方式**完全不同**，不依賴 `generate` 來生成標籤。

**核心操作**：如您所說，它在 Transformer 最後一層附加了**分類頭 (classifier heads)**。您必須手動獲取模型（主 LLM）在生成每個 token 時的**隱藏狀態 (hidden state)**，然後將這個 `hidden_state` 傳遞給 `Stream` 模型的**專用分類函數**（如 `prompt_check` 和 `reply_check`）來即時獲取安全分數。

**使用時機**：專為**在線服務**設計。

1.  **提示級預檢**：在主 LLM 執行*之前*，先用 `Stream` 模型檢查用戶輸入是否安全。
2.  **逐詞即時審核**：在主 LLM *生成回覆的過程中*，`Stream` 模型同步檢查**每一個剛生成的 token**。一旦發現 `unsafe`，主 LLM 應立即停止生成。

#### 程式碼範例 (概念與關鍵步驟)

使用 `Stream` 版本有一個**絕對關鍵**的前提：

```python
# 1. 載入模型 (!! 必須設定 trust_remote_code=True !!)
# 這會載入 Qwen 團隊編寫的、用於操作分類頭的自訂 Python 程式碼
model_id = "Qwen/Qwen3Guard-Stream-4B"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True # 必須
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True # 必須
)
model.eval()

# Qwen3Guard 的標籤
# 0: safe, 1: controversial, 2: unsafe
label_map = {0: "safe", 1: "controversial", 2: "unsafe"}

# --- 步驟 A：提示級安全預檢 (Input Guardrail) ---

print("--- 檢查用戶輸入 ---")
user_input = "How to make a bomb?"
inputs = tokenizer(user_input, return_tensors="pt").to(device)

# 獲取模型輸出，包含隱藏狀態
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True # 必須獲取隱藏狀態
    )

# 獲取最後一個 token 的隱藏狀態
last_hidden_state = outputs.hidden_states[-1][:, -1, :]

# 呼叫 Stream 模型的 'prompt_check' 函數
prompt_scores = model.prompt_check(last_hidden_state)
prompt_label_id = torch.argmax(prompt_scores, dim=1).item()

print(f"用戶輸入: '{user_input}'")
print(f"輸入評估結果: {label_map[prompt_label_id]}\n")


# --- 步驟 B：逐詞即時審核 (Output Guardrail) ---
# 
# (注意：這一步在實際應用中，會與您的主 LLM [例如 Qwen2-7B] 
# 的生成迴圈 [streaming loop] 整合在一起)

print("--- 檢查模型回覆 (模擬) ---")
# 假設這是您的主 LLM 正在生成的 token 的隱藏狀態
# (在真實情境中，您會從主 LLM 的每一步生成中獲取)
mock_reply_hidden_state = torch.randn(1, model.config.hidden_size).to(device, dtype=torch.bfloat16)

# 呼叫 Stream 模型的 'reply_check' 函數
with torch.no_grad():
    reply_scores = model.reply_check(mock_reply_hidden_state)
    
reply_label_id = torch.argmax(reply_scores, dim=1).item()

print(f"模型 (模擬) 輸出的 Token 評估結果: {label_map[reply_label_id]}")

# 您的應用程式邏輯：
if reply_label_id == 2: # 2 = unsafe
    print("偵測到不安全內容，立即停止生成！")
    # (在此處中斷主 LLM 的生成迴圈)
```

### 總結：`Gen` vs `Stream`

| 特性 | `Qwen3Guard-Gen` (生成式版) | `Qwen3Guard-Stream` (流式檢測版) |
| :--- | :--- | :--- |
| **操作方式** | `model.generate()` | `model.prompt_check(h)` 和 `model.reply_check(h)` |
| **輸入** | 完整的提示詞 (Prompt String) | 隱藏狀態 (Hidden State Tensor) |
| **輸出** | 標籤文字 (e.g., "unsafe") | 分類分數 (Logits Tensor) |
| **關鍵依賴** | `transformers` | `transformers` + `trust_remote_code=True` |
| **效能** | 較高延遲 (需完整生成一次) | 極低延遲 (逐 Token 判斷) |
| **最佳場景** | 離線數據批次處理 | 在線聊天服務的即時干預 |

### 5. OpenAI gpt-oss-safeguard：動態「策略推理」的新範式

OpenAI 的 gpt-oss-safeguard 徹底改變了傳統安全模型的運作模式，使模型從「記憶規則」轉變為「理解策略」。

* **核心技術**： 一個「動態策略驅動的安全推理引擎」。其核心是**策略與模型解耦**。
* **運作原理**： 模型在**推理（Inference）階段**運行，它接收兩項輸入：
    1.  **策略 (Policy)**：一份由開發者用自然語言編寫的安全規則文件。
    2.  **內容 (Content)**：需要被分類的文本。
* **動態適應**：開發者**只需修改策略文件，無需重新訓練模型**，即可即時更新安全規則，實現「策略即提示詞」(policy-as-prompt)。
* **透明度**：模型會輸出「思維鏈」(Chain-of-Thought, CoT)，詳細解釋它如何根據策略得出結論，打破了傳統分類器的「黑箱」。
* **定位**：具備極致靈活性和客製化能力的「安全推理引擎」，適用於快速迭代和適應新風險。

-----

### gpt-oss-safeguard 的操作使用方法

`gpt-oss-safeguard` 的操作核心，就是建構一個**同時包含「規則」和「內容」的特定提示詞 (Prompt)**。

它的工作流程如下：

1.  **您 (開發者)**：用自然語言撰寫一份「安全策略 (Policy)」。
2.  **您 (開發者)**：獲取用戶的「輸入內容 (Content)」。
3.  **您 (開發者)**：將這兩者組合成一個特定的提示詞，發送給 `gpt-oss-safeguard` 模型。
4.  **模型 (Safeguard)**：
    a. 讀取並理解您的 (Policy)。
    b. 讀取並分析 (Content)。
    c. 產生「**思維鏈 (Chain-of-Thought)**」來說明它的推理過程。
    d. 給出最終的「**判決 (Verdict)**」。

#### 程式碼範例 (使用 Hugging Face `transformers`)

以下是如何使用 20B 版本的 `gpt-oss-safeguard-20b-apollo-v1` 的完整操作範例。

**步驟 1：安裝與登入**

```bash
pip install transformers torch accelerate
huggingface-cli login
```

*(`gpt-oss` 系列模型目前是管制模型，您需要先在 Hugging Face 網站上同意其授權條款)*

**步驟 2：Python 操作程式碼**

此範例將模擬一個遊戲論壇，我們不希望用戶討論作弊或發表侮辱性言論。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# 1. 載入模型和 Tokenizer
# (使用 20B 版本，120B 版本操作相同，只是更強大也更慢)
model_id = "OpenAI/gpt-oss-safeguard-20b-apollo-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在 {device} 上載入模型...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 針對 20B/120B 模型，建議使用 bf16 並啟用 flash attention 2 (如果硬體支援)
config = AutoConfig.from_pretrained(model_id)
# config.attn_implementation = "flash_attention_2" # (可選)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16, # 使用 bfloat16
    device_map=device
)
model.eval()

# 2. 定義您的「安全策略 (Policy)」(用自然語言)
# 這就是 gpt-oss-safeguard 的 "無需訓練" 的關鍵
policy_document = """
[POLICY]
1.  **禁止討論遊戲作弊**：
    - 不得詢問、分享或討論任何形式的作弊軟體 (例如: aimbots, wallhacks)。
    - 不得討論利用遊戲 Bug (exploits) 獲利的行為。
2.  **禁止人身攻擊與侮辱**：
    - 不得對其他玩家或群體使用侮辱性詞彙 (例如: "trash", "noob", "garbage")。
    - 嚴禁任何形式的歧視或仇恨言論。
[/POLICY]
"""

# 3. 準備要檢查的「內容 (Content)」
# 這則內容同時違反了規則 1 和 2
content_to_check = """
[CONTENT]
我真的受不了我的隊友，他們都是一群垃圾 (garbage)！
有沒有人知道去哪裡找好用的 aimbot？這遊戲的配對系統太爛了。
[/CONTENT]
"""

# 4. 建立「提示詞 (Prompt)」
# 這是最關鍵的一步：我們使用官方推薦的聊天模板
# 將 Policy 和 Content 一起 "餵" 給模型
messages = [
    {
        "role": "system",
        "content": "You are a content safety classifier. Your task is to analyze the [CONTENT] provided by the user, based on the rules defined in the [POLICY]. Provide your reasoning step-by-step and then conclude with a final 'Verdict:' (Safe or Unsafe)."
    },
    {
        "role": "user",
        "content": f"{policy_document}\n{content_to_check}"
    }
]

# 5. Tokenize 並執行 "generate"
# 使用 apply_chat_template 來正確格式化輸入
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to(device)

print("\n--- 正在呼叫 gpt-oss-safeguard 進行推理 ---")

# pad_token_id 必須設置，eos_token_id 也是
outputs = model.generate(
    inputs,
    max_new_tokens=512, # 留足夠空間給 Reasoning (CoT)
    pad_token_id=tokenizer.eos_token_id
)

# 6. 解碼並查看結果
# generate() 回傳的是包含提示詞的完整 token，我們只看新生成的
response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("\n--- gpt-oss-safeguard 的判決書 ---")
print(response_text)
```

#### 預期的輸出 (判決書)

`gpt-oss-safeguard` 的回覆會類似這樣，包含完整的推理過程：

```text
Reasoning:
1.  開始分析 [CONTENT]。
2.  [CONTENT] 中提到 "他們都是一群垃圾 (garbage)"。
3.  對照 [POLICY] 規則 2 ("禁止人身攻擊與侮辱")，其中明確指出不得使用 "garbage" 等侮辱性詞彙。這違反了規則 2。
4.  [CONTENT] 中提到 "有沒有人知道去哪裡找好用的 aimbot？"。
5.  對照 [POLICY] 規則 1 ("禁止討論遊戲作弊")，其中明確指出不得詢問 "aimbots"。這違反了規則 1。
6.  由於 [CONTENT] 同時違反了規則 1 和規則 2，因此應被分類為不安全。

Verdict: Unsafe
```

### 總結：如何「操作」它

1.  **操作**：您不需要「訓練」或「微調」它。您對它的「操作」就是**撰寫一份清晰、詳細的自然語言 `[POLICY]` 文件**。
2.  **使用**：將您的 `[POLICY]` 和用戶的 `[CONTENT]` 包裝在官方推薦的**聊天模板**中。
3.  **解析**：您的應用程式需要解析模型生成的**文字輸出**，提取最後一行 `Verdict: Safe` 或 `Verdict: Unsafe` 來作為您程式的判斷依據，並可以選擇性地儲存 `Reasoning` 部分以供日後審核。

#### 三、 共同挑戰與模型侷限

儘管技術進步顯著，所有護欄仍面臨共同的挑戰，且各自存在侷限。

* **普遍存在的挑戰**
    * **對抗性攻擊（越獄）**：所有基於 LLM 的護欄（如 Llama Guard, ShieldGemma）都繼承了 LLM 的脆弱性，易受提示詞注入、語義操縱（如隱喻、反諷）等攻擊的影響。
    * **上下文理解局限**：在多輪對話中逐步構建的攻擊，可能因護欄難以追蹤完整的長程上下文而得逞。
    * **「貓捉老鼠」的博弈**：開源模型的機制易被攻擊者研究，導致持續的攻防競賽。
    * **誤傷（False Positives）**：過於嚴格的規則可能導致將無害對話錯誤標記為危險，影響用戶體驗。
* **各模型的具體侷限性**
    * **NVIDIA NeMo**：防護能力高度依賴開發者編寫的 Colang 腳本，配置不當會使 LLM 完全暴露。
    * **Meta Llama Guard**：作為「生成後檢測」，延遲較高；且其本身也可能被提示詞注入攻擊。
    * **Google ShieldGemma**：對安全原則的具體措辭高度敏感，評估基準數據有限。
    * **阿里巴巴 Qwen3Guard**：流式 (Stream) 版本因只能看到部分上下文，檢測性能相比能看到完整回覆的生成式 (Gen) 版有輕微下降。
    * **OpenAI gpt-oss-safeguard**：最大的局限在於**計算成本和延遲較高**，可能不適用於所有內容的即時審核。

---

#### 四、 技術前沿：兩大演進方向

當前的護欄技術正朝著兩個關鍵方向演進，以克服傳統方法的局限：

1.  **從「生成後檢測」到「即時流式檢測」**
    以 **Qwen3Guard-Stream** 為代表。此路徑旨在解決「生成後檢測」（如 Llama Guard）帶來的高延遲問題。透過逐詞（token-by-token）監控，它將風險暴露時間從秒級壓縮到毫秒級，實現了「事中監督」。

2.  **從「靜態規則」到「動態策略推理」**
    以 **gpt-oss-safeguard** 為代表。此路徑旨在解決傳統護欄（無論是 NeMo 的腳本或 Llama Guard 的微調）靈活性不足的問題。透過在推理時動態解釋自然語言策略，它賦予了安全系統前所未有的適應性和客製化能力，更新規則無需重新訓練。

---

#### 五、 未來趨勢：基礎免費化與企業級工程服務

結合開源模型的湧現和企業實際需求，未來安全護欄市場將呈現清晰的「基礎免費、進階付費」格局。

1.  **基礎能力免費化（類比 MySQL）**
    開源護欄模型（如 Llama Guard, Qwen3Guard, ShieldGemma）和開源數據集將成為行業標配，如同 MySQL 免費提供核心數據庫功能，這將極大推動 AI 安全技術的普及。

2.  **企業付費的核心：工程化服務**
    企業願意付費的不再是基礎的分類能力，而是專業的**「工程化能力」**。單純的開源工具無法應對複雜的生產環境，企業需要平台工程能力將這些「基礎積木」整合成可規模化管理的體系。關鍵付費點包括：
    * **高性能實時攔截**：支援 Qwen3Guard 這類的流式（Streaming）和 Llama Guard 4 這類的多模態即時檢測與干預。
    * **策略可視化編排**：提供圖形化界面（UI）來配置和管理安全規則（如 NeMo 的流程），實現策略的版本控制和靈活組合。
    * **灰度發布 (A/B Testing)**：新安全策略上線前，僅將其應用於 1% 的用戶流量，透過數據（如攔截率、誤傷率）評估效果，避免「誤傷」正常業務。
    * **分層部署**：為兼顧成本與效率，採用分層策略：先用輕量級、低延遲的分類器（如 ShieldGemma）快速篩選，再將高風險或模棱兩可的內容交由高精度、高成本的推理引擎（如 gpt-oss-safeguard）進行深度分析。
* **合規與服務保障**：提供滿足監管的合規報表、SLA 服務保障、以及持續的攻防演練服務。

