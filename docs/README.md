# Threads 內容策略分析報告

## 研究目標

透過數據分析找出 Threads 平台的關鍵成效因子，為中文創作者提供可操作的內容策略建議。

## 資料來源

| 項目 | 數值 |
|------|------|
| 分析期間 | 2024 - 2026 |
| 總貼文數 | ~9,000 篇 |
| 創作者數 | ~95 位 |
| 分析時區 | 台灣時間 (UTC+8) |
| 篩選條件 | 排除 >100k 粉絲的極端大帳號 |

## 分析報告

### [0. 資料集分布分析](./00-dataset-analysis.md)

- 粉絲數 / 貼文數分布與 Gini 係數
- 互動指標（按讚、回覆、轉發）的分布特徵
- 各指標間的相關性矩陣

### [1. 基礎指標分析](./01-basic-analysis.md)

- 發文時段分析：小時、時段、商業時段
- 內容形式分析：文長區間、句子數量
- 各維度的互動成效比較

### [2. 詞彙分析](./02-word-analysis.md)

- 詞頻統計與文字雲
- 高互動詞彙識別
- 詞彙象限分析（明星詞 / 藍海詞 / 紅海詞 / 地雷詞）
- 依粉絲量級分層的差異分析

### [3. 文章分類](./03-article-classification.md)

- AI 自動分類：教學型 / 經驗分享 / 觀點評論 / 資源整理 / 問答互動 / 成果展示 / 生活記錄 / 其他
- 各類別互動成效排名
- 最佳發文時段 x 文章類型交叉分析
- 最佳文長 x 文章類型交叉分析

### [4. 創作者分類](./04-creator-classification.md)

- AI 自動分類：知識教學型 / 創業歷程型 / 專業服務型 / 電商導購型 / 思想觀點型 / 品牌組織型 / 混合型
- 各類型互動成效與內容組合比較
- 發文頻率 vs. 互動成效分析
- 粉絲量級 x 創作者類型交叉分析

## 重現分析

```bash
# 0. 資料集分布
python scripts/dataset/analysis.py

# 1. 基礎指標
python scripts/basic/analysis.py

# 2. 詞彙分析
python scripts/word/analysis.py

# 3. 文章分類（需要 Gemini API Key）
python scripts/article/analysis.py

# 4. 創作者分類（需要 Gemini API Key）
python scripts/creator/analysis.py
```

## 分析工具

- Python 3.12+
- pandas / matplotlib / seaborn
- 中文斷詞：jieba
- AI 分類：Google Gemini
- 文字雲：wordcloud
