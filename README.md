# Threads Scraper

這是一個用於爬取 Threads 使用者資料和貼文的 Python 工具集。包含兩個主要腳本：一個用於根據關鍵字搜尋使用者，另一個用於爬取特定使用者的詳細資料與貼文。

## 功能

*   **搜尋使用者 (`fetch_users.py`)**: 根據關鍵字搜尋 Threads 使用者，並將發現的使用者名稱儲存到列表。
*   **爬取貼文 (`fetch_posts.py`)**: 爬取指定使用者的個人檔案資訊（粉絲數、Bio 等）及其發佈的貼文內容（按讚數、回覆數等）。
*   **自動化**: 支援批次處理與自動滾動載入更多內容。
*   **Cookies 管理**: 支援儲存與載入 Cookies 以維持登入狀態。

## 安裝

1.  **Clone 專案**
2.  **安裝相依套件**

    ```bash
    pip install -r requirements.txt
    ```
    *(或者若您使用 uv 或 pdm 等工具，請參考 `pyproject.toml`)*
    
    或者手動安裝必要套件：
    ```bash
    pip install selenium beautifulsoup4 pydantic
    ```

3.  **ChromeDriver**: 確保您已安裝與 Chrome 瀏覽器版本相符的 ChromeDriver，並將其加入系統路徑，或放置於專案目錄下。

## 使用方法

### 1. 登入並取得 Cookies

首次使用建議先執行 `fetch_posts.py` 來進行登入並儲存 Cookies，以便後續腳本使用。

```bash
python fetch_posts.py
```
若無 `threads_cookies.pkl` 檔案，腳本會提示您手動登入。登入完成後在終端機按下 Enter，Cookies 將被儲存。

### 2. 搜尋使用者

使用 `fetch_users.py` 來根據關鍵字搜尋使用者。

```bash
python fetch_users.py
```
*   預設關鍵字為：`["創業", "行銷", "ㄧ人公司"]` (可在程式碼中修改)
*   結果將儲存於 `discovered_users.json`。

### 3. 爬取使用者貼文

**單一使用者爬取**:
直接在指令後方加上使用者名稱或個人檔案網址。

```bash
python fetch_posts.py [username]
# 或
python fetch_posts.py https://www.threads.net/@username
```

**批次爬取**:
若不帶參數執行，且目錄下存在 `discovered_users.json`，腳本將自動讀取列表並批次爬取。

```bash
python fetch_posts.py
```
*   資料將儲存於 `data/[username]/[timestamp].json`。
*   若資料夾中已存在該使用者的資料，批次模式預設會跳過。

## 輸出資料結構

資料以 JSON 格式儲存，包含 `profile` 與 `posts` 兩部分：

```json
{
  "profile": {
    "username": "example_user",
    "full_name": "Example Name",
    "followers": 1000,
    "bio": "User bio...",
    "bio_links": ["..."]
  },
  "posts": [
    {
      "id": "123456...",
      "text": "Post content...",
      "url": "https://www.threads.net/post/...",
      "posted_at": "2023-12-01T12:00:00.000Z",
      "like_count": 100,
      "reply_count": 10,
      "repost_count": 5,
      "forward_count": 0
    }
  ]
}
```

## 注意事項

*   請勿過度頻繁使用以避免觸發 Threads 的反爬蟲機制。
*   `threads_cookies.pkl` 包含您的登入資訊，請妥善保管，勿將其加入版控。

