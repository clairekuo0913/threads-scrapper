import os
import platform
import sys

import jieba
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import load_data, remove_extreme_outliers, set_chinese_font


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
            "每個",
            "這個",
            "那個",
            "這些",
            "那些",
            "這種",
            "那種",
            "因為",
            "所以",
            "但是",
            "而且",
            "而是",
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
            "常常",
            "總是",
            "完全",
            "真正",
            "大家",
            "今天",
            "現在",
            "時候",
            "開始",
            "之後",
            "之前",
            "後來",
            "最近",
            "什麼",
            "怎麼",
            "為什麼",
            "如何",
            "如果",
            "沒有",
            "不是",
            "不要",
            "不能",
            "不會",
            "不用",
            "只有",
            "一定",
            "已經",
            "自己",
            "我們",
            "他們",
            "你們",
            "有人",
            "別人",
            "知道",
            "看到",
            "出來",
            "進來",
            "回來",
            "起來",
            "下去",
            "上去",
            "文章",
            "內容",
            "東西",
            "事情",
            "時間",
            "地方",
            "方式",
            "問題",
            "結果",
            "這樣",
            "那樣",
            "一起",
            "到底",
            "是不是",
            "不再",
            "我當",
            "Ctrl",
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
            "...",
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
            "10",
            "已靜音",
            # 數字類（日期、時間、價格等噪音）
            "00",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "50",
            "80",
            "90",
            "100",
            "200",
            "300",
            "500",
            "1000",
            "2000",
            "2024",
            "2025",
            "2026",
            # 連接詞/填充詞（無實質意義）
            "一樣",
            "還有",
            "只要",
            "一點",
            "以前",
            "這麼",
            "所有",
            "那麼",
            "正在",
            "甚至",
            "其他",
            "好像",
            "任何",
            "這裡",
            "特別",
            "每次",
            "不管",
            "各種",
            "除了",
            "必須",
            "能夠",
            "這邊",
            "通常",
            "大概",
            "認為",
            "感覺",
            "繼續",
            "這次",
            "那次",
            "上次",
            "下次",
            "為了",
            "通過",
            "然後",
            # 標點符號類
            "..",
            "....",
            "--",
            "──",
            # 短代碼/URL
            "tw",
            "https",
            "http",
            "www",
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
    # Drop rows where words is NaN
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
    Now includes distinct creator count for each word.
    """
    print("\n--- Word Engagement Analysis ---")

    # Group by word with distinct creator count
    word_stats = word_df.groupby("words").agg(
        {
            "like_count": ["count", "median", "mean"],
            "reply_count": ["median", "mean"],
            "repost_count": ["median", "mean"],
            "username": "nunique",  # Distinct creators using this word
        }
    )

    # Flatten columns
    word_stats.columns = ["_".join(col).strip() for col in word_stats.columns.values]
    word_stats = word_stats.rename(
        columns={"like_count_count": "count", "username_nunique": "distinct_creators"}
    )

    # Filter by min_count
    word_stats = word_stats[word_stats["count"] >= min_count]

    # Sort by median likes
    top_engagement = word_stats.sort_values("like_count_median", ascending=False)

    print(
        f"\nTop 20 High Engagement Words (Min {min_count} occurrences, sorted by Median Likes):"
    )
    print(
        top_engagement[
            ["count", "distinct_creators", "like_count_median", "like_count_mean"]
        ].head(20)
    )

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
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    # Group by category
    cat_stats = cat_df.groupby("category")[
        ["like_count", "reply_count", "repost_count"]
    ].mean()
    cat_counts = cat_df["category"].value_counts()

    print("\nCategory Performance:")
    print(cat_stats)

    return cat_stats, cat_counts, cat_df


def analyze_quadrant(
    word_stats: pd.DataFrame, word_counts: pd.Series, output_dir: str
) -> pd.DataFrame:
    """
    Analyze words in a 2x2 quadrant:
    - High Freq + High Engagement = Star Words (明星詞)
    - High Freq + Low Engagement = Red Ocean Words (紅海詞)
    - Low Freq + High Engagement = Blue Ocean Words (藍海詞)
    - Low Freq + Low Engagement = Avoid (避開)
    """
    print("\n--- Quadrant Analysis ---")

    # Merge frequency and engagement data
    quadrant_df = word_stats.copy()
    quadrant_df["frequency"] = quadrant_df.index.map(word_counts)

    # Calculate thresholds (median of each metric)
    freq_threshold = quadrant_df["frequency"].median()
    engagement_threshold = quadrant_df["like_count_median"].median()

    print(f"Frequency threshold (median): {freq_threshold:.0f}")
    print(f"Engagement threshold (median): {engagement_threshold:.0f}")

    # Classify into quadrants
    def classify_quadrant(row):
        high_freq = row["frequency"] >= freq_threshold
        high_eng = row["like_count_median"] >= engagement_threshold

        if high_freq and high_eng:
            return "Star (明星詞)"
        elif high_freq and not high_eng:
            return "Red Ocean (紅海詞)"
        elif not high_freq and high_eng:
            return "Blue Ocean (藍海詞)"
        else:
            return "Avoid (避開)"

    quadrant_df["quadrant"] = quadrant_df.apply(classify_quadrant, axis=1)

    # Print summary
    print("\nQuadrant Distribution:")
    print(quadrant_df["quadrant"].value_counts())

    print("\nTop 5 Star Words (明星詞 - 高頻高互動):")
    stars = quadrant_df[quadrant_df["quadrant"] == "Star (明星詞)"].nlargest(
        5, "like_count_median"
    )
    if not stars.empty:
        print(
            stars[["frequency", "distinct_creators", "like_count_median"]].to_string()
        )

    print("\nTop 5 Blue Ocean Words (藍海詞 - 低頻高互動):")
    blue_ocean = quadrant_df[quadrant_df["quadrant"] == "Blue Ocean (藍海詞)"].nlargest(
        5, "like_count_median"
    )
    if not blue_ocean.empty:
        print(
            blue_ocean[
                ["frequency", "distinct_creators", "like_count_median"]
            ].to_string()
        )

    print("\nTop 5 Red Ocean Words (紅海詞 - 高頻低互動，內容同質化):")
    red_ocean = quadrant_df[quadrant_df["quadrant"] == "Red Ocean (紅海詞)"].nlargest(
        5, "frequency"
    )
    if not red_ocean.empty:
        print(
            red_ocean[
                ["frequency", "distinct_creators", "like_count_median"]
            ].to_string()
        )

    # Save quadrant data
    quadrant_csv = os.path.join(output_dir, "quadrant_analysis.csv")
    os.makedirs(output_dir, exist_ok=True)
    quadrant_df.to_csv(quadrant_csv)
    print(f"\nSaved quadrant analysis to {quadrant_csv}")

    return quadrant_df


def plot_word_analysis(
    word_counts: pd.Series,
    word_stats: pd.DataFrame,
    cat_stats: pd.DataFrame,
    cat_df: pd.DataFrame,
    output_dir: str,
    quadrant_df: pd.DataFrame = None,
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
    if not word_counts.empty:
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
    if not word_counts.empty:
        plt.figure(figsize=(12, 8))
        # Sort ascending for horizontal bar chart (top items at top)
        word_counts.head(20).sort_values().plot(kind="barh")
        plt.title("Top 20 Frequent Words")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_words_freq.png"))
        plt.close()

    # 3. Top 20 High Engagement Words
    if not word_stats.empty:
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

    # 5. Quadrant Plot (Frequency vs Engagement)
    if quadrant_df is not None and not quadrant_df.empty:
        plt.figure(figsize=(14, 10))

        # Color mapping for quadrants
        colors = {
            "Star (明星詞)": "#2ecc71",  # Green
            "Blue Ocean (藍海詞)": "#3498db",  # Blue
            "Red Ocean (紅海詞)": "#e74c3c",  # Red
            "Avoid (避開)": "#95a5a6",  # Gray
        }

        for quadrant, color in colors.items():
            mask = quadrant_df["quadrant"] == quadrant
            subset = quadrant_df[mask]
            if not subset.empty:
                plt.scatter(
                    subset["frequency"],
                    subset["like_count_median"],
                    c=color,
                    label=quadrant,
                    alpha=0.6,
                    s=50,
                )

        # Add labels for top words in each quadrant
        for quadrant in ["Star (明星詞)", "Blue Ocean (藍海詞)"]:
            mask = quadrant_df["quadrant"] == quadrant
            top_words = quadrant_df[mask].nlargest(3, "like_count_median")
            for word, row in top_words.iterrows():
                plt.annotate(
                    word,
                    (row["frequency"], row["like_count_median"]),
                    fontsize=8,
                    alpha=0.8,
                )

        # Draw threshold lines
        freq_threshold = quadrant_df["frequency"].median()
        eng_threshold = quadrant_df["like_count_median"].median()
        plt.axvline(x=freq_threshold, color="gray", linestyle="--", alpha=0.5)
        plt.axhline(y=eng_threshold, color="gray", linestyle="--", alpha=0.5)

        plt.xlabel("Frequency (出現次數)")
        plt.ylabel("Median Likes (中位數按讚)")
        plt.title("Word Quadrant Analysis (詞彙四象限)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "word_quadrant.png"))
        plt.close()

    print(f"Plots saved to {output_dir}")


def get_follower_tier(followers):
    if followers <= 100:
        return "0-100"
    elif followers <= 1000:
        return "100-1k"
    elif followers <= 10000:
        return "1k-10k"
    elif followers <= 100000:
        return "10k-100k"
    elif followers <= 1000000:
        return "100k-1m"
    else:
        return "1m+"


if __name__ == "__main__":
    # Test setup
    stop_words = setup_jieba()
    print("Jieba setup complete.")

    # Path relative to this script (scripts/word/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(script_dir)  # scripts/
    data_dir = os.path.join(scripts_dir, "../data")

    print(f"Loading data from {data_dir}...")
    df = load_data(data_dir)

    if df.empty:
        print("No data loaded.")
    else:
        print(f"Loaded {len(df)} posts.")

        # Add follower tier
        # Ensure followers column exists and is numeric
        if "followers" not in df.columns:
            df["followers"] = 0
        df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)

        # Remove extreme outliers (viral posts with > 1000 likes)
        df = remove_extreme_outliers(df, column="like_count", threshold=1000)

        df["tier"] = df["followers"].apply(get_follower_tier)

        # Define tier order
        tier_order = ["0-100", "100-1k", "1k-10k", "10k-100k", "100k-1m", "1m+"]
        available_tiers = df["tier"].unique().tolist()
        tiers_to_process = ["All"] + [t for t in tier_order if t in available_tiers]

        # Sampling settings
        SAMPLE_TIERS = ["100-1k", "1k-10k", "10k-100k"]
        MAX_POSTS_PER_USER = {
            "100-1k": 50,
            "1k-10k": 50,
            "10k-100k": 100,
        }

        for tier in tiers_to_process:
            print(f"\n=== Processing Tier: {tier} ===")

            if tier == "All":
                current_df = df.copy()
            else:
                current_df = df[df["tier"] == tier].copy()

            if current_df.empty:
                print(f"No posts for tier {tier}, skipping.")
                continue

            # Apply sampling for specific tiers to reduce spam/high-frequency bias
            if tier in SAMPLE_TIERS:
                limit = MAX_POSTS_PER_USER[tier]
                original_count = len(current_df)
                original_users = current_df["username"].nunique()
                print(
                    f"Filtering users with <= {limit} posts and sampling max {limit} posts..."
                )

                # Filter users who have enough posts
                user_counts = current_df["username"].value_counts()
                active_users = user_counts[user_counts > limit].index
                current_df = current_df[current_df["username"].isin(active_users)]

                # Sort by date (newest first) to get the most relevant recent content
                current_df = current_df.sort_values("posted_at", ascending=False)
                # Group by username and take top N
                current_df = current_df.groupby("username").head(limit)
                sampled_users = current_df["username"].nunique()
                print(
                    f"Reduced from {original_count} posts ({original_users} users) to {len(current_df)} posts ({sampled_users} users)."
                )

            print(f"Analyzing {len(current_df)} posts for tier {tier}...")

            # 1. Frequency
            word_df, word_counts = analyze_word_frequency(current_df, stop_words)

            if word_df.empty:
                print("No words found.")
                continue

            # 2. Engagement
            word_stats = analyze_word_engagement(word_df)

            # 3. Categories
            cat_stats, cat_counts, cat_df = analyze_categories(word_df)

            # 4. Quadrant Analysis
            plots_dir = os.path.join(scripts_dir, f"outputs/plots/word-analysis/{tier}")
            quadrant_df = analyze_quadrant(word_stats, word_counts, plots_dir)

            # 5. Save Data
            output_csv = os.path.join(
                scripts_dir, f"outputs/tables/word-analysis-{tier}.csv"
            )
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            word_stats.to_csv(output_csv)
            print(f"Saved word stats to {output_csv}")

            # 6. Plots
            plot_word_analysis(
                word_counts, word_stats, cat_stats, cat_df, plots_dir, quadrant_df
            )
