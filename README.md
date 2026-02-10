# Threads Scraper & Analysis

Threads 社群平台的爬蟲與數據分析工具。從使用者搜尋、貼文爬取到多維度內容分析，完整涵蓋資料蒐集與洞察產出的流程。

## 功能概覽

| 模組 | 說明 |
|------|------|
| **使用者搜尋** | 依關鍵字搜尋 Threads 使用者，建立目標名單 |
| **貼文爬取** | 爬取個人檔案與歷史貼文，結構化儲存 |
| **資料集分析** | 粉絲 / 貼文分布、Gini 係數、相關性矩陣 |
| **基礎指標分析** | 時段、文長、句數 vs. 互動成效 |
| **詞彙分析** | 斷詞詞頻、高互動詞彙、象限分析、文字雲 |
| **文章分類** | AI 自動分類 8 種文章類型，分析各類成效 |
| **創作者分類** | AI 自動分類 7 種創作者類型，比較成功模式 |

## 安裝

需要 Python 3.12+ 與 Chrome 瀏覽器。

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -e .
```

> ChromeDriver 須與 Chrome 版本相符。macOS ARM64 版本已包含於專案中。

## 使用方法

### 1. 登入取得 Cookies

首次執行會開啟瀏覽器供手動登入，登入後按 Enter 儲存 Cookies。後續執行自動載入。

```bash
python fetch_posts.py
```

### 2. 搜尋使用者

依關鍵字搜尋目標使用者，結果存入 `data/discovered_users.json`。

```bash
python fetch_users.py
```

### 3. 爬取貼文

```bash
# 爬取單一使用者
python fetch_posts.py <username>
python fetch_posts.py https://www.threads.net/@username

# 批次爬取（讀取 discovered_users.json）
python fetch_posts.py
```

資料存放於 `data/<username>/<timestamp>.json`，結構如下：

```json
{
  "profile": {
    "username": "example",
    "full_name": "Example Name",
    "followers": 1000,
    "bio": "...",
    "bio_links": []
  },
  "posts": [
    {
      "id": "...",
      "text": "貼文內容",
      "url": "https://www.threads.net/post/...",
      "posted_at": "2025-01-01T12:00:00.000Z",
      "like_count": 50,
      "reply_count": 3,
      "repost_count": 2,
      "forward_count": 1
    }
  ]
}
```

### 4. 執行分析

```bash
# 資料集分布分析
python scripts/dataset/analysis.py

# 基礎指標分析（時段、文長、互動）
python scripts/basic/analysis.py

# 詞彙分析（詞頻、文字雲、象限圖）
python scripts/word/analysis.py

# 文章分類（AI 分類 + 成效分析）
python scripts/article/analysis.py

# 創作者分類（AI 分類 + 模式比較）
python scripts/creator/analysis.py
```

分析結果輸出至 `scripts/outputs/`：

```
scripts/outputs/
├── plots/       # 視覺化圖表
├── tables/      # CSV 數據表
└── cache/       # AI 分類快取
```

## 技術架構

```
fetch_users.py          # 關鍵字搜尋使用者
fetch_posts.py          # 爬取個人檔案與貼文
scripts/
├── shared/utils.py     # 共用工具（資料載入、中文字型、文本清洗）
├── dataset/            # 資料集分布分析
├── basic/              # 基礎指標分析
├── word/               # 詞彙分析（jieba 斷詞）
├── article/            # 文章分類（Google Gemini）
└── creator/            # 創作者分類（Google Gemini）
docs/                   # 分析報告文件（GitBook）
```

**關鍵技術：**

- **爬蟲**：Selenium + BeautifulSoup，Cookie 驗證、動態滾動載入、自動去重
- **資料驗證**：Pydantic 結構化模型
- **中文處理**：jieba 斷詞 + 自訂詞典 / 停用詞
- **AI 分類**：Google Gemini，含快取避免重複呼叫
- **視覺化**：matplotlib + seaborn，支援中文字型
- **分層分析**：依粉絲量級分層（0-100 / 100-1k / 1k-10k / 10k+）

## 環境變數

複製 `.env.example` 並填入所需金鑰：

```bash
cp .env.example .env
```

AI 分類功能（文章 / 創作者分類）需要設定 Google Gemini API Key。

## 注意事項

- `data/` 目錄已加入 `.gitignore`，包含使用者隱私資料
- `threads_cookies.pkl` 包含登入憑證，請勿提交至版控
- 請勿過度頻繁爬取，以避免觸發 Threads 反爬蟲機制
