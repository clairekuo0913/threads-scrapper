"""
Article Classification Analysis using Google Gemini.

Classifies Threads posts into 8 categories using AI,
then performs cross-analysis on engagement, timing, length, and keywords.
"""

import json
import os
import sys
import time
from datetime import timedelta, timezone

import google.generativeai as genai
import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import load_data, remove_extreme_outliers, set_chinese_font

# Load environment variables
load_dotenv()

# Define timezone (UTC+8 for Taiwan)
TZ_TW = timezone(timedelta(hours=8))

# ── Category definitions ──────────────────────────────────────────────────────

CATEGORIES = [
    "教學型",
    "經驗分享",
    "觀點評論",
    "資源整理",
    "問答互動",
    "成果展示",
    "生活記錄",
    "其他",
]

CLASSIFICATION_PROMPT = """你是一個專業的社群媒體內容分類專家。請將以下每篇文章分類到指定類別之一，並給出信心分數。

類別定義：
1. 教學型 - 有步驟、方法的教學內容（例：「3步驟學會XX」、「如何做到XX」）
2. 經驗分享 - 個人經歷、案例分享（例：「我如何從0到1」、「我的創業故事」）
3. 觀點評論 - 思考、評論、觀點（例：「關於XX的思考」、「我認為XX」）
4. 資源整理 - 工具清單、資源推薦（例：「10個好用工具」、「推薦XX」）
5. 問答互動 - 提問、徵求意見、討論（例：「大家怎麼看？」、「請問XX」）
6. 成果展示 - 里程碑、成就展示（例：「突破10萬粉」、「月營收XX」）
7. 生活記錄 - 日常心情、隨筆（例：「今天的感悟」、「週末日記」）
8. 其他 - 無法明確分類的內容

請分類以下 {count} 篇文章，以 JSON 陣列格式回傳：

{posts}

回傳格式（嚴格遵守，不要加任何其他文字）：
[
  {{"category": "類別名稱", "confidence": 0.95}},
  {{"category": "類別名稱", "confidence": 0.85}},
  ...
]

注意：
- 每篇文章對應一個分類結果，順序必須一致
- confidence 為 0-1 之間的浮點數
- category 必須是上述 8 個類別之一
- 如果文章太短或無法判斷，分類為「其他」
"""


# ── Gemini classification ─────────────────────────────────────────────────────


def init_gemini() -> genai.GenerativeModel:
    """Initialize Gemini model with API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in .env file or environment."
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


def classify_posts_batch(texts: list[str], model: genai.GenerativeModel) -> list[dict]:
    """
    Classify a batch of posts using Gemini.

    Args:
        texts: List of post texts to classify.
        model: Initialized Gemini model.

    Returns:
        List of dicts with 'category' and 'confidence' keys.
    """
    # Format posts for the prompt
    formatted_posts = ""
    for i, text in enumerate(texts):
        # Truncate very long posts to save tokens
        truncated = text[:500] if len(text) > 500 else text
        # Escape any curly braces in the text
        truncated = truncated.replace("{", "{{").replace("}", "}}")
        formatted_posts += f"文章 {i + 1}：{truncated}\n\n"

    prompt = CLASSIFICATION_PROMPT.format(count=len(texts), posts=formatted_posts)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

        # Parse JSON response
        result = json.loads(response.text)

        # Validate result length
        if len(result) != len(texts):
            print(
                f"  Warning: Expected {len(texts)} results, got {len(result)}. Padding with '其他'."
            )
            while len(result) < len(texts):
                result.append({"category": "其他", "confidence": 0.0})
            result = result[: len(texts)]

        # Validate categories
        for item in result:
            if item.get("category") not in CATEGORIES:
                item["category"] = "其他"
            if item.get("confidence", 0) < 0.75:
                item["category"] = "其他"

        return result

    except Exception as e:
        print(f"  Error classifying batch: {e}")
        return [{"category": "其他", "confidence": 0.0} for _ in texts]


def load_or_classify(
    df: pd.DataFrame, cache_path: str, model: genai.GenerativeModel, batch_size: int = 5
) -> pd.DataFrame:
    """
    Classify posts, using cache to avoid re-classifying.

    Args:
        df: DataFrame with posts (must have 'id' and 'text' columns).
        cache_path: Path to JSON cache file.
        model: Initialized Gemini model.
        batch_size: Number of posts per API call.

    Returns:
        DataFrame with added 'category' and 'confidence' columns.
    """
    # Load existing cache
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached classifications.")

    # Find posts that need classification
    df = df.copy()
    needs_classification = df[~df["id"].astype(str).isin(cache.keys())]

    print(
        f"Total posts: {len(df)}, Cached: {len(df) - len(needs_classification)}, "
        f"Need classification: {len(needs_classification)}"
    )

    if len(needs_classification) > 0:
        # Process in batches
        total_batches = (len(needs_classification) + batch_size - 1) // batch_size
        print(f"Processing {total_batches} batches (batch_size={batch_size})...")

        for batch_idx in range(0, len(needs_classification), batch_size):
            batch = needs_classification.iloc[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            if batch_num % 50 == 0 or batch_num == 1:
                print(f"  Batch {batch_num}/{total_batches}...")

            texts = batch["text"].tolist()
            ids = batch["id"].astype(str).tolist()

            results = classify_posts_batch(texts, model)

            # Save to cache
            for post_id, result in zip(ids, results):
                cache[post_id] = result

            # Save cache periodically (every 100 batches)
            if batch_num % 100 == 0:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                print(f"  Cache saved ({len(cache)} entries).")

            # Rate limiting
            time.sleep(0.5)

        # Final cache save
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Classification complete. Cache saved ({len(cache)} entries).")

    # Merge classifications into DataFrame
    df["category"] = (
        df["id"].astype(str).map(lambda x: cache.get(x, {}).get("category", "其他"))
    )
    df["confidence"] = (
        df["id"].astype(str).map(lambda x: cache.get(x, {}).get("confidence", 0.0))
    )

    return df


# ── Analysis functions ────────────────────────────────────────────────────────


def analyze_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze post distribution across categories."""
    print("\n--- Category Distribution ---")

    dist = (
        df.groupby("category")
        .agg(
            post_count=("id", "count"),
            avg_likes=("like_count", "mean"),
            avg_replies=("reply_count", "mean"),
            avg_reposts=("repost_count", "mean"),
            distinct_creators=("username", "nunique"),
        )
        .round(1)
    )

    dist["pct"] = (dist["post_count"] / dist["post_count"].sum() * 100).round(1)
    dist = dist.sort_values("post_count", ascending=False)

    print(dist.to_string())
    return dist


def analyze_category_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze engagement metrics per category with median and mean."""
    print("\n--- Category Engagement ---")

    engagement = (
        df.groupby("category")
        .agg(
            post_count=("id", "count"),
            like_mean=("like_count", "mean"),
            like_median=("like_count", "median"),
            reply_mean=("reply_count", "mean"),
            reply_median=("reply_count", "median"),
            repost_mean=("repost_count", "mean"),
            repost_median=("repost_count", "median"),
        )
        .round(1)
    )

    # Rank by mean likes
    engagement = engagement.sort_values("like_mean", ascending=False)

    print("Engagement by Category (sorted by avg likes):")
    print(engagement.to_string())
    return engagement


def analyze_category_time(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze best posting times for each category."""
    print("\n--- Category x Time Analysis ---")

    df = df.copy()
    df["hour"] = df["posted_at_tw"].dt.hour
    df["weekday"] = df["posted_at_tw"].dt.dayofweek  # 0=Monday, 6=Sunday

    # Create time slot mapping
    def get_time_slot(hour):
        if 7 <= hour < 9:
            return "通勤時間 (7-9)"
        elif 9 <= hour < 12:
            return "上午 (9-12)"
        elif 12 <= hour < 14:
            return "午休 (12-14)"
        elif 14 <= hour < 18:
            return "下午 (14-18)"
        elif 18 <= hour < 20:
            return "晚間 (18-20)"
        elif 20 <= hour < 23:
            return "黃金時段 (20-23)"
        else:
            return "深夜 (23-7)"

    df["time_slot"] = df["hour"].apply(get_time_slot)

    # Category x Time Slot pivot - mean likes
    pivot = df.pivot_table(
        values="like_count",
        index="category",
        columns="time_slot",
        aggfunc="mean",
    ).round(1)

    # Reorder columns by time
    slot_order = [
        "通勤時間 (7-9)",
        "上午 (9-12)",
        "午休 (12-14)",
        "下午 (14-18)",
        "晚間 (18-20)",
        "黃金時段 (20-23)",
        "深夜 (23-7)",
    ]
    existing_slots = [s for s in slot_order if s in pivot.columns]
    pivot = pivot[existing_slots]

    print("Average Likes by Category x Time Slot:")
    print(pivot.to_string())

    # Find best time for each category
    best_times = pivot.idxmax(axis=1)
    print("\nBest Time Slot per Category:")
    for cat, slot in best_times.items():
        print(f"  {cat}: {slot}")

    return pivot


def analyze_category_length(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze optimal text length for each category."""
    print("\n--- Category x Text Length ---")

    df = df.copy()
    df["text_len"] = df["text"].str.len()
    df["sentence_count"] = df["text"].str.count(r"[。！？\n]") + 1

    # Length bins
    length_bins = [0, 50, 100, 200, 300, 500, 1000, float("inf")]
    length_labels = [
        "<50字",
        "50-100字",
        "100-200字",
        "200-300字",
        "300-500字",
        "500-1000字",
        ">1000字",
    ]
    df["length_bin"] = pd.cut(df["text_len"], bins=length_bins, labels=length_labels)

    # Category x Length - mean likes
    pivot = df.pivot_table(
        values="like_count",
        index="category",
        columns="length_bin",
        aggfunc="mean",
        observed=False,
    ).round(1)

    print("Average Likes by Category x Text Length:")
    print(pivot.to_string())

    # Best length per category
    best_lengths = pivot.idxmax(axis=1)
    print("\nBest Length Bin per Category:")
    for cat, length in best_lengths.items():
        print(f"  {cat}: {length}")

    # Summary stats per category
    length_summary = (
        df.groupby("category")
        .agg(
            avg_text_len=("text_len", "mean"),
            median_text_len=("text_len", "median"),
            avg_sentences=("sentence_count", "mean"),
            median_sentences=("sentence_count", "median"),
        )
        .round(1)
    )

    print("\nText Length Summary per Category:")
    print(length_summary.to_string())

    return pivot


def setup_jieba_for_keywords():
    """Minimal jieba setup for keyword extraction (reuse from word analysis)."""
    custom_words = [
        "一人公司",
        "自媒體",
        "流量",
        "變現",
        "被動收入",
        "創新創業",
        "商業模式",
        "獲利",
        "演算法",
        "互動率",
        "個人品牌",
        "斜槓",
        "副業",
        "遠端工作",
        "數位遊牧",
        "AI",
        "ChatGPT",
        "SaaS",
        "電子報",
        "Notion",
        "Threads",
        "Instagram",
        "IG",
        "FB",
        "Facebook",
    ]
    for word in custom_words:
        jieba.add_word(word)

    stop_words = set(
        [
            "的",
            "了",
            "是",
            "我",
            "你",
            "他",
            "在",
            "也",
            "就",
            "有",
            "和",
            "與",
            "及",
            "或",
            "這",
            "那",
            "都",
            "會",
            "要",
            "去",
            "一個",
            "一下",
            "一些",
            "每個",
            "這個",
            "那個",
            "這些",
            "那些",
            "因為",
            "所以",
            "但是",
            "而且",
            "或是",
            "還是",
            "只是",
            "不過",
            "雖然",
            "其實",
            "真的",
            "覺得",
            "可以",
            "可能",
            "應該",
            "就是",
            "非常",
            "很多",
            "比較",
            "一直",
            "大家",
            "今天",
            "現在",
            "時候",
            "開始",
            "之後",
            "之前",
            "什麼",
            "怎麼",
            "如何",
            "如果",
            "沒有",
            "不是",
            "不要",
            "不能",
            "不會",
            "只有",
            "一定",
            "已經",
            "自己",
            "我們",
            "他們",
            "你們",
            "知道",
            "看到",
            "文章",
            "內容",
            "東西",
            "事情",
            "時間",
            "方式",
            "問題",
            "結果",
            "這樣",
            "一起",
            "然後",
            "為了",
            "感覺",
            "認為",
            "繼續",
            "通常",
            "還有",
            "所有",
            "其他",
            "特別",
            "各種",
            "除了",
            "必須",
            "能夠",
            "\n",
            " ",
            "，",
            "。",
            "！",
            "？",
            "、",
            "：",
            "；",
            "（",
            "）",
            "「",
            "」",
            "『",
            "』",
            "…",
            "—",
            "～",
            "...",
            "已靜音",
            "https",
            "http",
            "www",
            "com",
            "tw",
            "net",
        ]
    )
    return stop_words


def analyze_category_keywords(df: pd.DataFrame, stop_words: set) -> dict:
    """Analyze top keywords for each category using jieba."""
    print("\n--- Category x Keywords ---")

    results = {}
    for category in CATEGORIES:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue

        # Tokenize all texts in this category
        all_words = []
        for text in cat_df["text"]:
            if not isinstance(text, str):
                continue
            words = jieba.cut(text)
            all_words.extend(
                [w for w in words if w not in stop_words and len(w.strip()) > 1]
            )

        if not all_words:
            continue

        word_counts = pd.Series(all_words).value_counts()
        top_words = word_counts.head(10)

        results[category] = top_words
        print(f"\n{category} Top 10 Keywords:")
        print(top_words.to_string())

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_all(
    df: pd.DataFrame,
    dist_df: pd.DataFrame,
    engagement_df: pd.DataFrame,
    time_pivot: pd.DataFrame,
    length_pivot: pd.DataFrame,
    keywords: dict,
    plots_dir: str,
) -> None:
    """Generate all visualizations for article classification analysis."""
    os.makedirs(plots_dir, exist_ok=True)
    set_chinese_font()

    # 1. Category Distribution (pie + bar)
    _plot_category_distribution(dist_df, plots_dir)

    # 2. Category Engagement (grouped bar)
    _plot_category_engagement(engagement_df, plots_dir)

    # 3. Time x Category Heatmap
    _plot_category_time_heatmap(time_pivot, plots_dir)

    # 4. Length x Category Heatmap
    _plot_category_length(length_pivot, plots_dir)

    # 5. Keywords by Category
    _plot_category_keywords(keywords, plots_dir)

    print(f"\nAll plots saved to {plots_dir}")


def _plot_category_distribution(dist_df: pd.DataFrame, plots_dir: str):
    """Plot category distribution as horizontal bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: bar chart with post counts
    colors = sns.color_palette("Set2", len(dist_df))
    dist_sorted = dist_df.sort_values("post_count", ascending=True)
    bars = axes[0].barh(dist_sorted.index, dist_sorted["post_count"], color=colors)
    axes[0].set_xlabel("文章數")
    axes[0].set_title("各類別文章數量")
    # Add value labels
    for bar, pct in zip(bars, dist_sorted["pct"]):
        axes[0].text(
            bar.get_width() + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.0f} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    # Right: average likes per category
    eng_sorted = dist_df.sort_values("avg_likes", ascending=True)
    bars2 = axes[1].barh(eng_sorted.index, eng_sorted["avg_likes"], color="orange")
    axes[1].set_xlabel("平均按讚數")
    axes[1].set_title("各類別平均按讚數")
    for bar in bars2:
        axes[1].text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "category_distribution.png"), dpi=150)
    plt.close()
    print("  Saved category_distribution.png")


def _plot_category_engagement(engagement_df: pd.DataFrame, plots_dir: str):
    """Plot engagement metrics comparison across categories."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    metrics = [
        ("like_mean", "平均按讚數", "#e74c3c"),
        ("reply_mean", "平均回覆數", "#3498db"),
        ("repost_mean", "平均轉發數", "#2ecc71"),
    ]

    for ax, (col, title, color) in zip(axes, metrics):
        sorted_df = engagement_df.sort_values(col, ascending=True)
        bars = ax.barh(sorted_df.index, sorted_df[col], color=color, alpha=0.8)
        ax.set_xlabel(title)
        ax.set_title(title)
        for bar in bars:
            ax.text(
                bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "category_engagement.png"), dpi=150)
    plt.close()
    print("  Saved category_engagement.png")


def _plot_category_time_heatmap(time_pivot: pd.DataFrame, plots_dir: str):
    """Plot heatmap of category x time slot engagement."""
    if time_pivot.empty:
        return

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        time_pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "平均按讚數"},
    )
    plt.title("各類別 × 時段 平均按讚數")
    plt.xlabel("時段")
    plt.ylabel("文章類別")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "category_time_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved category_time_heatmap.png")


def _plot_category_length(length_pivot: pd.DataFrame, plots_dir: str):
    """Plot heatmap of category x text length engagement."""
    if length_pivot.empty:
        return

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        length_pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "平均按讚數"},
    )
    plt.title("各類別 × 文章長度 平均按讚數")
    plt.xlabel("文章長度")
    plt.ylabel("文章類別")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "category_length.png"), dpi=150)
    plt.close()
    print("  Saved category_length.png")


def _plot_category_keywords(keywords: dict, plots_dir: str):
    """Plot top keywords for each category in a grid."""
    cats_with_data = {k: v for k, v in keywords.items() if len(v) > 0}
    if not cats_with_data:
        return

    n_cats = len(cats_with_data)
    cols = min(n_cats, 4)
    rows = (n_cats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_cats == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors_list = sns.color_palette("Set2", n_cats)

    for idx, (cat, word_counts) in enumerate(cats_with_data.items()):
        ax = axes[idx]
        top5 = word_counts.head(5).sort_values()
        ax.barh(top5.index, top5.values, color=colors_list[idx])
        ax.set_title(cat, fontsize=12, fontweight="bold")
        ax.set_xlabel("出現次數")

    # Hide unused axes
    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("各類別 Top 5 關鍵詞", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "category_keywords.png"), dpi=150)
    plt.close()
    print("  Saved category_keywords.png")


# ── Main entry point ──────────────────────────────────────────────────────────


def main():
    # Setup paths (scripts/article/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(script_dir)  # scripts/
    data_dir = os.path.join(scripts_dir, "../data")
    cache_path = os.path.join(scripts_dir, "outputs/cache/article-classifications.json")
    output_dir = os.path.join(scripts_dir, "outputs/tables")
    plots_dir = os.path.join(scripts_dir, "outputs/plots/article-classification")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load data
    print("=" * 60)
    print("Article Classification Analysis")
    print("=" * 60)

    print(f"\nLoading data from {data_dir}...")
    df = load_data(data_dir)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(df)} posts from {df['username'].nunique()} creators.")

    # Ensure followers is numeric
    if "followers" not in df.columns:
        df["followers"] = 0
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)

    # Filter: exclude large accounts (>100k) and extreme outliers
    df = df[df["followers"] <= 100000].copy()
    df = remove_extreme_outliers(df, column="like_count", threshold=1000)

    # Filter out empty texts
    df = df[df["text"].str.strip().str.len() > 0].copy()

    print(
        f"\nAfter filtering: {len(df)} posts from {df['username'].nunique()} creators."
    )

    # 2. Classify with Gemini
    print("\n" + "=" * 60)
    print("Phase 1: AI Classification (Gemini)")
    print("=" * 60)

    model = init_gemini()
    df = load_or_classify(df, cache_path, model, batch_size=5)

    # Print classification summary
    print("\nClassification Summary:")
    print(df["category"].value_counts().to_string())

    # 3. Run analysis
    print("\n" + "=" * 60)
    print("Phase 2: Analysis")
    print("=" * 60)

    # 3a. Distribution
    dist_df = analyze_category_distribution(df)

    # 3b. Engagement
    engagement_df = analyze_category_engagement(df)

    # 3c. Time x Category
    time_pivot = analyze_category_time(df)

    # 3d. Length x Category
    length_pivot = analyze_category_length(df)

    # 3e. Keywords x Category
    stop_words = setup_jieba_for_keywords()
    keywords = analyze_category_keywords(df, stop_words)

    # 4. Generate plots
    print("\n" + "=" * 60)
    print("Phase 3: Visualization")
    print("=" * 60)

    plot_all(df, dist_df, engagement_df, time_pivot, length_pivot, keywords, plots_dir)

    # 5. Export CSV
    print("\n" + "=" * 60)
    print("Phase 4: Export")
    print("=" * 60)

    # Main classification results
    export_cols = [
        "id",
        "username",
        "followers",
        "text",
        "posted_at_tw",
        "like_count",
        "reply_count",
        "repost_count",
        "category",
        "confidence",
    ]
    existing_cols = [c for c in export_cols if c in df.columns]
    export_df = df[existing_cols].copy()
    export_path = os.path.join(output_dir, "article-classification.csv")
    export_df.to_csv(export_path, index=False, encoding="utf-8-sig")
    print(f"Saved classification results to {export_path}")

    # Summary table
    summary_path = os.path.join(output_dir, "article-classification-summary.csv")
    dist_df.to_csv(summary_path, encoding="utf-8-sig")
    print(f"Saved summary to {summary_path}")

    # Engagement table
    engagement_path = os.path.join(output_dir, "article-classification-engagement.csv")
    engagement_df.to_csv(engagement_path, encoding="utf-8-sig")
    print(f"Saved engagement to {engagement_path}")

    print("\n" + "=" * 60)
    print("Done! All outputs saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
