# Threads 分析報告

## 研究目標

找出關鍵成效因子，優化內容策略

## 分析維度

### [1. 基礎指標分析](./01-basic-analysis.md)

- 時間維度：小時、時段、商業時段
- 內容形式：文長、句子數
- 成效指標：按讚、回覆、轉發

### [2. 詞彙分析](./02-word-analysis.md) 🚧

- 詞頻統計：最常提到的關鍵詞
- 高互動詞彙：哪些詞帶來最多互動
- 詞彙分類：依主題分類分析

### [3. 文章分類](./03-article-classification.md) 🚧

- AI 自動分類
- 各類別成效分析
- 最佳主題識別

### [4. 創作者分類](./04-creator-classification.md) 🚧

- 創作者類型標記
- 各類型成效比較
- 成功模式識別

## 資料來源

- **分析期間**：2024-2026
- **總貼文數**：9,071 篇
- **創作者數**：96 位
- **分析時區**：台灣時間 (UTC+8)

## 使用方式

```bash
# 1. 基礎分析
python scripts/basic_analysis.py

# 2. 詞彙分析
python scripts/word_analysis.py

# 3. 文章分類 (待開發)
python scripts/article_classifier.py

# 4. 創作者分類 (待開發)
python scripts/creator_classifier.py
```

## 分析工具

- Python 3.11+
- pandas, matplotlib
- 中文斷詞：jieba / ckiptagger
- AI 分類：OpenAI GPT-4 / Anthropic Claude

---

📊 **最後更新**：2026-01-21
