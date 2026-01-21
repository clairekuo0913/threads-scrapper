import os
import platform

import jieba
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data, set_chinese_font
from wordcloud import WordCloud


def setup_jieba():
    """
    Initialize jieba with custom dictionary and stop words.
    """
    # Add custom dictionary words
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
        "Notion",
        "AI",
        "ChatGPT",
        "SaaS",
        "電子報",
        "個人品牌",
        "斜槓",
        "副業",
        "遠端工作",
        "數位遊牧",
        "Threads",
        "Instagram",
        "IG",
        "FB",
        "Facebook",
    ]
    for word in custom_words:
        jieba.add_word(word)

    # Define stop words
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
            "https",
            "http",
            "com",
            "www",
            "t",
            "co",
            "instagram",
            "threads",
            "net",
            "org",
            "html",
            "htm",
            "一個",
            "一下",
            "一些",
            "因為",
            "所以",
            "但是",
            "而且",
            "其實",
            "真的",
            "覺得",
            "可以",
            "可能",
            "應該",
            "就是",
            "大家",
            "今天",
            "現在",
            "時候",
            "開始",
            "只是",
            "還是",
            "什麼",
            "怎麼",
            "為什麼",
            "如何",
            "比較",
            "非常",
            "很多",
            "自己",
            "我們",
            "他們",
            "你們",
            "文章",
            "內容",
            "知道",
            "看到",
            "沒有",
            "不是",
            "一定",
            "已經",
            "不過",
            "雖然",
            "之後",
            "之前",
            "後來",
            "最近",
            "一直",
            "這樣",
            "那樣",
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
            "“",
            "”",
            "’",
            "‘",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
        ]
    )
    return stop_words


def segment_text(text: str, stop_words: list[str]):
    """
    Segment text into words using jieba, filtering out stop words.
    """
    if not isinstance(text, str):
        return []
    # Use cut_for_search for better recall or cut for precision? Default cut is fine.
    words = jieba.cut(text)
    # Filter: not in stop_words, length > 1 (ignore single chars usually noise), strip whitespace
    return [w for w in words if w not in stop_words and len(w.strip()) > 1]


def analyze_word_frequency(
    df: pd.DataFrame, stop_words: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Analyze word frequency.
    Returns word_df (exploded) and word_counts.
    """
    print("\n--- Word Frequency Analysis ---")

    # Segment words
    # This might take a while for large datasets
    df["words"] = df["text"].apply(lambda x: segment_text(x, stop_words))

    # Explode to get one row per word
    # This creates a much larger dataframe
    word_df = df.explode("words")
    # Drop rows where words is NaN (empty lists resulted in NaN after explode? No, empty list explodes to nothing/NaN depending on pandas version/settings, usually rows disappear or become NaN. Let's check dropna)
    word_df = word_df.dropna(subset=["words"])

    # Count frequency
    word_counts = word_df["words"].value_counts()
    print("\nTop 30 Frequent Words:")
    print(word_counts.head(30))

    return word_df, word_counts


def analyze_word_engagement(word_df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """
    Analyze engagement metrics for each word.
    Uses median to handle outliers.
    """
    print("\n--- Word Engagement Analysis ---")

    # Group by word
    # Calculate count, median (robust), mean (sensitive)
    word_stats = word_df.groupby("words")[
        ["like_count", "reply_count", "repost_count"]
    ].agg(["count", "median", "mean"])

    # Flatten columns
    word_stats.columns = ["_".join(col).strip() for col in word_stats.columns.values]
    # The count is the same for all metrics, just take one
    word_stats = word_stats.rename(columns={"like_count_count": "count"})
    # Drop redundant count columns if any (pandas agg might produce like_count_count, reply_count_count etc if we asked for it, but here we asked agg on list of cols. Wait, syntax above: groupby(cols).agg(['count', ...]) applies count to all cols. So we get like_count_count, reply_count_count. They are identical.)
    # Let's keep one count and drop others.
    word_stats = word_stats.drop(
        columns=["reply_count_count", "repost_count_count"], errors="ignore"
    )

    # Filter by min_count
    word_stats = word_stats[word_stats["count"] >= min_count]

    # Sort by median likes
    top_engagement = word_stats.sort_values("like_count_median", ascending=False)

    print(
        f"\nTop 20 High Engagement Words (Min {min_count} occurrences, sorted by Median Likes):"
    )
    print(top_engagement[["count", "like_count_median", "like_count_mean"]].head(20))

    return word_stats


def analyze_categories(word_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Analyze performance by word category.
    """
    print("\n--- Category Analysis ---")

    # Define categories
    categories = {
        "Business": [
            "創業",
            "公司",
            "營收",
            "商業",
            "獲利",
            "變現",
            "賺錢",
            "客戶",
            "產品",
            "銷售",
            "市場",
            "品牌",
            "行銷",
            "SaaS",
            "B2B",
            "B2C",
        ],
        "Media": [
            "流量",
            "粉絲",
            "互動",
            "演算法",
            "觸及",
            "自媒體",
            "內容",
            "經營",
            "IG",
            "Threads",
            "Reels",
            "貼文",
            "漲粉",
        ],
        "Tech": [
            "AI",
            "ChatGPT",
            "工具",
            "軟體",
            "程式",
            "代碼",
            "自動化",
            "效率",
            "Notion",
            "Python",
            "No-code",
            "網站",
            "App",
        ],
        "Growth": [
            "學習",
            "成長",
            "思考",
            "習慣",
            "目標",
            "改變",
            "心態",
            "進步",
            "閱讀",
            "書籍",
            "知識",
            "技能",
        ],
        "Career": [
            "工作",
            "職涯",
            "面試",
            "履歷",
            "薪水",
            "離職",
            "轉職",
            "主管",
            "同事",
            "公司",
            "遠端",
            "自由接案",
        ],
        "Life": [
            "生活",
            "休息",
            "焦慮",
            "壓力",
            "快樂",
            "健康",
            "運動",
            "睡眠",
            "旅行",
            "美食",
            "朋友",
            "家人",
        ],
        "Money": [
            "理財",
            "投資",
            "股票",
            "被動收入",
            "資產",
            "財富",
            "存錢",
            "省錢",
            "副業",
            "斜槓",
        ],
    }

    # Invert mapping: word -> category
    word_to_cat = {}
    for cat, words in categories.items():
        for word in words:
            word_to_cat[word] = cat

    # Map words to categories
    # Use .copy() to avoid SettingWithCopyWarning if word_df is a view
    word_df = word_df.copy()
    word_df["category"] = word_df["words"].map(word_to_cat)

    # Filter only categorized words
    cat_df = word_df.dropna(subset=["category"])

    if cat_df.empty:
        print("No categorized words found.")
        return pd.DataFrame(), pd.Series()

    # Group by category
    cat_stats = cat_df.groupby("category")[
        ["like_count", "reply_count", "repost_count"]
    ].mean()
    cat_counts = cat_df["category"].value_counts()

    print("\nCategory Performance:")
    print(cat_stats)

    return cat_stats, cat_counts, cat_df


def plot_word_analysis(
    word_counts: pd.Series,
    word_stats: pd.DataFrame,
    cat_stats: pd.DataFrame,
    cat_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Generate plots for word analysis.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_chinese_font()

    # 1. Word Cloud
    print("Generating Word Cloud...")

    # Find a font path for WordCloud (requires actual path, not family name)
    font_path = None
    system = platform.system()
    if system == "Darwin":
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        for c in candidates:
            if os.path.exists(c):
                font_path = c
                break
    elif system == "Windows":
        candidates = ["C:\\Windows\\Fonts\\msjh.ttc", "C:\\Windows\\Fonts\\simhei.ttf"]
        for c in candidates:
            if os.path.exists(c):
                font_path = c
                break
    elif system == "Linux":
        candidates = ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]
        for c in candidates:
            if os.path.exists(c):
                font_path = c
                break

    wc_kwargs = {
        "width": 1200,
        "height": 600,
        "background_color": "white",
        "max_words": 200,
    }
    if font_path:
        wc_kwargs["font_path"] = font_path

    # Overall Word Cloud
    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(word_counts)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wordcloud.png"))
    plt.close()

    # Category Word Clouds
    # Map internal category names to filename suffixes
    cat_map = {
        "Business": "business",
        "Media": "media",
        "Tech": "tools",  # Mapping "Tech" to "tools" as per doc requirement
    }

    for cat_name, file_suffix in cat_map.items():
        print(f"Generating Word Cloud for {cat_name}...")
        # Filter words for this category
        cat_words = cat_df[cat_df["category"] == cat_name]["words"]

        if cat_words.empty:
            print(f"No words found for category {cat_name}, skipping.")
            continue

        cat_word_counts = cat_words.value_counts()

        wc_cat = WordCloud(**wc_kwargs)
        wc_cat.generate_from_frequencies(cat_word_counts)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc_cat, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {cat_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"wordcloud_{file_suffix}.png"))
        plt.close()

    # 2. Top 20 Words by Frequency
    plt.figure(figsize=(12, 8))
    # Sort ascending for horizontal bar chart (top items at top)
    word_counts.head(20).sort_values().plot(kind="barh")
    plt.title("Top 20 Frequent Words")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_words_freq.png"))
    plt.close()

    # 3. Top 20 High Engagement Words
    plt.figure(figsize=(12, 8))
    top_eng = word_stats.sort_values("like_count_median", ascending=False).head(20)
    top_eng["like_count_median"].sort_values().plot(kind="barh", color="orange")
    plt.title("Top 20 High Engagement Words (Median Likes)")
    plt.xlabel("Median Likes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_words_engagement.png"))
    plt.close()

    # 4. Category Performance
    if not cat_stats.empty:
        plt.figure(figsize=(12, 6))
        cat_stats["like_count"].sort_values().plot(kind="bar", color="purple")
        plt.title("Average Likes by Category")
        plt.ylabel("Average Likes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "category_performance.png"))
        plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    # Test setup
    stop_words = setup_jieba()
    print("Jieba setup complete.")

    # Path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")

    print(f"Loading data from {data_dir}...")
    df = load_data(data_dir)

    if df.empty:
        print("No data loaded.")
    else:
        print(f"Loaded {len(df)} posts.")

        # 1. Frequency
        word_df, word_counts = analyze_word_frequency(df, stop_words)

        # 2. Engagement
        word_stats = analyze_word_engagement(word_df)

        # 3. Categories
        cat_stats, cat_counts, cat_df = analyze_categories(word_df)

        # 4. Save Data
        output_csv = os.path.join(script_dir, "tables/word-analysis.csv")
        word_stats.to_csv(output_csv)
        print(f"Saved word stats to {output_csv}")

        # 5. Plots
        plots_dir = os.path.join(script_dir, "plots/word-analysis")
        plot_word_analysis(word_counts, word_stats, cat_stats, cat_df, plots_dir)
