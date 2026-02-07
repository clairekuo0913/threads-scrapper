"""
Creator Classification Analysis using Google Gemini.

Classifies Threads creators into 7 business-model types using AI,
then analyzes success patterns, engagement, content mix, and posting frequency.
"""

import glob
import json
import os
import sys
import time
from collections import Counter
from datetime import timedelta, timezone

import google.generativeai as genai
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

TZ_TW = timezone(timedelta(hours=8))

# ── Creator type definitions ──────────────────────────────────────────────────

CREATOR_TYPES = [
    "知識教學型",
    "創業歷程型",
    "專業服務型",
    "電商導購型",
    "思想觀點型",
    "品牌組織型",
    "混合型",
]

CREATOR_PROMPT = """你是一個專業的社群媒體創作者分析專家。請根據以下創作者的完整資料，判斷此創作者屬於哪種類型。

創作者類型定義：
1. 知識教學型 - 教技能、方法、工具，賣線上課程或教學服務（特徵：步驟化教學、課程連結、教學系列）
2. 創業歷程型 - 記錄創業/轉職旅程，分享個人故事與成長（特徵：個人敘事、轉折故事、building in public）
3. 專業服務型 - 用專業技能接案/服務客戶，用內容獲客（特徵：領域專業tips、客戶案例、預約CTA）
4. 電商導購型 - 賣實體商品，用 Threads 做銷售導流（特徵：商品照、價格、促銷、預購、下單）
5. 思想觀點型 - 分享觀點、趨勢分析、思想領袖（特徵：評論、熱門議題、產業洞察）
6. 品牌組織型 - 公司/組織帳號，非個人品牌（特徵：活動公告、品牌推廣、組織訊息）
7. 混合型 - 無明確主導風格，內容多元混合

── 創作者資料 ──

帳號名稱：{username}
顯示名稱：{full_name}
粉絲數：{followers}
個人簡介：{bio}
Bio 連結數：{bio_links_count}

── 數據統計 ──

貼文數：{post_count}
平均按讚：{avg_likes:.1f}
中位數按讚：{median_likes:.1f}
最高按讚：{max_likes}
平均回覆：{avg_replies:.1f}
平均轉發：{avg_reposts:.1f}
每週發文：{posts_per_week:.1f} 篇
{content_mix_str}

── 代表性貼文（按讚數最高的 5 篇）──

{sample_posts}

── 請回傳 JSON ──

回傳格式（嚴格遵守，不要加任何其他文字）：
{{"primary_type": "類型名稱", "secondary_type": "次要類型或null", "confidence": 0.9, "reasoning": "簡短分類理由"}}

注意：
- primary_type 必須是上述 7 個類型之一
- secondary_type 可以是 7 個類型之一或 null
- confidence 為 0-1 之間的浮點數
"""


# ── Phase 1: Feature extraction ───────────────────────────────────────────────


def load_profiles(data_dir: str) -> dict[str, dict]:
    """
    Load profile data (bio, full_name, bio_links) from raw JSON files.
    Returns dict keyed by username.
    """
    profiles = {}
    pattern = os.path.join(data_dir, "*", "*.json")
    files = glob.glob(pattern)

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = data.get("profile", {})
            username = profile.get("username", "unknown")
            if username not in profiles:
                profiles[username] = {
                    "username": username,
                    "full_name": profile.get("full_name", ""),
                    "bio": profile.get("bio", ""),
                    "followers": profile.get("followers", 0),
                    "bio_links_count": len(profile.get("bio_links", [])),
                }
        except Exception:
            continue

    print(f"Loaded {len(profiles)} creator profiles.")
    return profiles


def load_article_cache(cache_path: str) -> dict:
    """Load article classification cache from 03."""
    if not os.path.exists(cache_path):
        print(f"Article classification cache not found at {cache_path}. Skipping content mix.")
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Loaded article cache with {len(cache)} entries.")
    return cache


def build_creator_features(
    df: pd.DataFrame,
    profiles: dict[str, dict],
    article_cache: dict,
) -> pd.DataFrame:
    """
    Build a creator-level feature DataFrame.
    """
    print("\n--- Building Creator Features ---")

    creators = []

    for username, group in df.groupby("username"):
        profile = profiles.get(username, {})

        # Basic profile
        row = {
            "username": username,
            "full_name": profile.get("full_name", ""),
            "bio": profile.get("bio", ""),
            "followers": profile.get("followers", group["followers"].iloc[0]),
            "bio_links_count": profile.get("bio_links_count", 0),
        }

        # Post aggregation
        row["post_count"] = len(group)
        row["avg_text_len"] = group["text"].str.len().mean()
        row["avg_sentence_count"] = (group["text"].str.count(r"[。！？\n]") + 1).mean()
        row["avg_likes"] = group["like_count"].mean()
        row["median_likes"] = group["like_count"].median()
        row["max_likes"] = group["like_count"].max()
        row["avg_replies"] = group["reply_count"].mean()
        row["avg_reposts"] = group["repost_count"].mean()

        # Time features
        if "posted_at_tw" in group.columns and not group["posted_at_tw"].isna().all():
            valid_times = group["posted_at_tw"].dropna()
            if len(valid_times) > 0:
                hours = valid_times.dt.hour
                row["preferred_hour"] = int(hours.mode().iloc[0]) if len(hours.mode()) > 0 else 12
                date_range = (valid_times.max() - valid_times.min()).days
                row["posting_span_days"] = max(date_range, 1)
                row["posts_per_week"] = row["post_count"] / max(date_range / 7, 1)
            else:
                row["preferred_hour"] = 12
                row["posting_span_days"] = 1
                row["posts_per_week"] = 0
        else:
            row["preferred_hour"] = 12
            row["posting_span_days"] = 1
            row["posts_per_week"] = 0

        # Content mix from article classification cache
        if article_cache:
            post_ids = group["id"].astype(str).tolist()
            categories_for_user = []
            for pid in post_ids:
                entry = article_cache.get(pid, {})
                cat = entry.get("category")
                if cat:
                    categories_for_user.append(cat)

            if categories_for_user:
                cat_counter = Counter(categories_for_user)
                total = len(categories_for_user)
                content_mix = {
                    cat: round(count / total, 3)
                    for cat, count in cat_counter.most_common()
                }
                row["content_mix"] = json.dumps(content_mix, ensure_ascii=False)
                row["dominant_article_category"] = cat_counter.most_common(1)[0][0]
            else:
                row["content_mix"] = "{}"
                row["dominant_article_category"] = "N/A"
        else:
            row["content_mix"] = "{}"
            row["dominant_article_category"] = "N/A"

        # Store top 5 posts by likes for Gemini prompt
        top_posts = group.nlargest(5, "like_count")
        sample_texts = []
        for _, post in top_posts.iterrows():
            text = str(post["text"])[:300]
            sample_texts.append(
                f"[讚:{post['like_count']} 回:{post['reply_count']}] {text}"
            )
        row["sample_posts"] = "\n\n".join(sample_texts)

        creators.append(row)

    creators_df = pd.DataFrame(creators)
    print(f"Built features for {len(creators_df)} creators.")
    return creators_df


# ── Phase 2: Gemini classification ────────────────────────────────────────────


def init_gemini() -> genai.GenerativeModel:
    """Initialize Gemini model with API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in .env file or environment."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def classify_single_creator(
    row: pd.Series, model: genai.GenerativeModel
) -> dict:
    """Classify a single creator using Gemini."""
    content_mix_str = ""
    if row.get("content_mix") and row["content_mix"] != "{}":
        try:
            mix = json.loads(row["content_mix"])
            parts = [f"  {cat}: {pct:.0%}" for cat, pct in mix.items()]
            content_mix_str = "文章類別分布：\n" + "\n".join(parts)
        except Exception:
            pass

    prompt = CREATOR_PROMPT.format(
        username=row["username"],
        full_name=row.get("full_name", ""),
        followers=row.get("followers", 0),
        bio=row.get("bio", ""),
        bio_links_count=row.get("bio_links_count", 0),
        post_count=row.get("post_count", 0),
        avg_likes=row.get("avg_likes", 0),
        median_likes=row.get("median_likes", 0),
        max_likes=row.get("max_likes", 0),
        avg_replies=row.get("avg_replies", 0),
        avg_reposts=row.get("avg_reposts", 0),
        posts_per_week=row.get("posts_per_week", 0),
        content_mix_str=content_mix_str,
        sample_posts=row.get("sample_posts", "（無貼文）"),
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        result = json.loads(response.text)

        # Validate
        if result.get("primary_type") not in CREATOR_TYPES:
            result["primary_type"] = "混合型"
        if result.get("secondary_type") and result["secondary_type"] not in CREATOR_TYPES:
            result["secondary_type"] = None

        return result

    except Exception as e:
        print(f"  Error classifying {row['username']}: {e}")
        return {
            "primary_type": "混合型",
            "secondary_type": None,
            "confidence": 0.0,
            "reasoning": f"Error: {e}",
        }


def load_or_classify_creators(
    creators_df: pd.DataFrame,
    cache_path: str,
    model: genai.GenerativeModel,
) -> pd.DataFrame:
    """Classify all creators, using cache to avoid re-classifying."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached creator classifications.")

    creators_df = creators_df.copy()
    needs_classification = creators_df[
        ~creators_df["username"].isin(cache.keys())
    ]

    print(
        f"Total creators: {len(creators_df)}, Cached: {len(creators_df) - len(needs_classification)}, "
        f"Need classification: {len(needs_classification)}"
    )

    if len(needs_classification) > 0:
        for idx, (_, row) in enumerate(needs_classification.iterrows()):
            username = row["username"]
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Classifying {idx + 1}/{len(needs_classification)}: {username}")

            result = classify_single_creator(row, model)
            cache[username] = result

            # Save cache every 20 creators
            if (idx + 1) % 20 == 0:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

            time.sleep(0.5)

        # Final save
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Classification complete. Cache saved ({len(cache)} entries).")

    # Merge into DataFrame
    creators_df["primary_type"] = creators_df["username"].map(
        lambda u: cache.get(u, {}).get("primary_type", "混合型")
    )
    creators_df["secondary_type"] = creators_df["username"].map(
        lambda u: cache.get(u, {}).get("secondary_type")
    )
    creators_df["confidence"] = creators_df["username"].map(
        lambda u: cache.get(u, {}).get("confidence", 0.0)
    )
    creators_df["reasoning"] = creators_df["username"].map(
        lambda u: cache.get(u, {}).get("reasoning", "")
    )

    return creators_df


# ── Phase 3: Analysis functions ───────────────────────────────────────────────


def get_follower_tier(followers):
    if followers <= 100:
        return "0-100"
    elif followers <= 1000:
        return "100-1k"
    elif followers <= 10000:
        return "1k-10k"
    elif followers <= 100000:
        return "10k-100k"
    else:
        return "100k+"


def analyze_type_distribution(creators_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze creator type distribution."""
    print("\n--- Creator Type Distribution ---")

    dist = (
        creators_df.groupby("primary_type")
        .agg(
            creator_count=("username", "count"),
            avg_followers=("followers", "mean"),
            total_posts=("post_count", "sum"),
            avg_posts_per_creator=("post_count", "mean"),
        )
        .round(1)
    )
    dist["pct"] = (dist["creator_count"] / dist["creator_count"].sum() * 100).round(1)
    dist = dist.sort_values("creator_count", ascending=False)

    print(dist.to_string())
    return dist


def analyze_type_engagement(creators_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze engagement metrics per creator type."""
    print("\n--- Creator Type Engagement ---")

    engagement = (
        creators_df.groupby("primary_type")
        .agg(
            creator_count=("username", "count"),
            avg_likes=("avg_likes", "mean"),
            median_likes=("median_likes", "mean"),
            avg_replies=("avg_replies", "mean"),
            avg_reposts=("avg_reposts", "mean"),
            avg_followers=("followers", "mean"),
        )
        .round(1)
    )
    engagement = engagement.sort_values("avg_likes", ascending=False)

    print(engagement.to_string())
    return engagement


def analyze_type_by_tier(creators_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of creator type x follower tier."""
    print("\n--- Creator Type x Follower Tier ---")

    creators_df = creators_df.copy()
    creators_df["tier"] = creators_df["followers"].apply(get_follower_tier)

    tier_order = ["0-100", "100-1k", "1k-10k", "10k-100k", "100k+"]
    cross = pd.crosstab(
        creators_df["primary_type"],
        creators_df["tier"],
    )
    # Reorder columns
    existing = [t for t in tier_order if t in cross.columns]
    cross = cross[existing]

    print(cross.to_string())
    return cross


def analyze_top_creators(creators_df: pd.DataFrame) -> pd.DataFrame:
    """Top 3 creators per type by avg likes."""
    print("\n--- Top Creators per Type ---")

    rows = []
    for ctype in CREATOR_TYPES:
        type_df = creators_df[creators_df["primary_type"] == ctype]
        if type_df.empty:
            continue
        top3 = type_df.nlargest(3, "avg_likes")
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            rows.append({
                "type": ctype,
                "rank": rank,
                "username": row["username"],
                "followers": row["followers"],
                "avg_likes": round(row["avg_likes"], 1),
                "post_count": row["post_count"],
                "bio": str(row.get("bio", ""))[:60],
            })

    top_df = pd.DataFrame(rows)
    if not top_df.empty:
        print(top_df.to_string(index=False))
    return top_df


def analyze_content_mix_by_type(creators_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze article category distribution per creator type."""
    print("\n--- Content Mix by Creator Type ---")

    rows = []
    for ctype in CREATOR_TYPES:
        type_df = creators_df[creators_df["primary_type"] == ctype]
        if type_df.empty:
            continue

        # Aggregate content mix across creators of this type
        combined = Counter()
        total = 0
        for _, row in type_df.iterrows():
            try:
                mix = json.loads(row.get("content_mix", "{}"))
                for cat, pct in mix.items():
                    combined[cat] += pct
                total += 1
            except Exception:
                continue

        if total > 0:
            avg_mix = {cat: round(val / total, 3) for cat, val in combined.items()}
            avg_mix["creator_type"] = ctype
            rows.append(avg_mix)

    if not rows:
        return pd.DataFrame()

    mix_df = pd.DataFrame(rows).set_index("creator_type").fillna(0)
    print(mix_df.round(2).to_string())
    return mix_df


def analyze_posting_frequency(creators_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze posting frequency vs engagement by type."""
    print("\n--- Posting Frequency Analysis ---")

    freq = (
        creators_df.groupby("primary_type")
        .agg(
            avg_posts_per_week=("posts_per_week", "mean"),
            median_posts_per_week=("posts_per_week", "median"),
            avg_likes=("avg_likes", "mean"),
            creator_count=("username", "count"),
        )
        .round(2)
    )
    freq = freq.sort_values("avg_posts_per_week", ascending=False)

    print(freq.to_string())
    return freq


# ── Phase 4: Plots ────────────────────────────────────────────────────────────


def plot_all(
    creators_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    engagement_df: pd.DataFrame,
    tier_cross: pd.DataFrame,
    top_df: pd.DataFrame,
    mix_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    plots_dir: str,
) -> None:
    """Generate all visualizations."""
    os.makedirs(plots_dir, exist_ok=True)
    set_chinese_font()

    _plot_distribution(dist_df, plots_dir)
    _plot_performance(engagement_df, plots_dir)
    _plot_content_mix(mix_df, plots_dir)
    _plot_frequency(creators_df, plots_dir)
    _plot_engagement_by_tier(creators_df, plots_dir)
    _plot_top_creators(top_df, plots_dir)

    print(f"\nAll plots saved to {plots_dir}")


def _plot_distribution(dist_df: pd.DataFrame, plots_dir: str):
    """Creator type distribution bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: creator counts
    dist_sorted = dist_df.sort_values("creator_count", ascending=True)
    colors = sns.color_palette("Set2", len(dist_sorted))
    bars = axes[0].barh(dist_sorted.index, dist_sorted["creator_count"], color=colors)
    axes[0].set_xlabel("創作者數")
    axes[0].set_title("各類型創作者數量")
    for bar, pct in zip(bars, dist_sorted["pct"]):
        axes[0].text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.0f} ({pct:.1f}%)",
            va="center", fontsize=9,
        )

    # Right: avg followers per type
    fol_sorted = dist_df.sort_values("avg_followers", ascending=True)
    bars2 = axes[1].barh(fol_sorted.index, fol_sorted["avg_followers"], color="steelblue")
    axes[1].set_xlabel("平均粉絲數")
    axes[1].set_title("各類型平均粉絲數")
    for bar in bars2:
        axes[1].text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():,.0f}",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "creator_distribution.png"), dpi=150)
    plt.close()
    print("  Saved creator_distribution.png")


def _plot_performance(engagement_df: pd.DataFrame, plots_dir: str):
    """Engagement metrics comparison (3-panel)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    metrics = [
        ("avg_likes", "平均按讚數", "#e74c3c"),
        ("avg_replies", "平均回覆數", "#3498db"),
        ("avg_reposts", "平均轉發數", "#2ecc71"),
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
                va="center", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "creator_performance.png"), dpi=150)
    plt.close()
    print("  Saved creator_performance.png")


def _plot_content_mix(mix_df: pd.DataFrame, plots_dir: str):
    """Stacked bar chart of article types per creator type."""
    if mix_df.empty:
        print("  Skipping content mix plot (no data).")
        return

    # Select top article categories for readability
    top_cats = mix_df.sum().nlargest(6).index.tolist()
    plot_df = mix_df[top_cats].copy()
    plot_df["其他類別"] = mix_df.drop(columns=top_cats, errors="ignore").sum(axis=1)

    ax = plot_df.plot(kind="barh", stacked=True, figsize=(14, 8), colormap="Set3")
    ax.set_xlabel("比例")
    ax.set_title("各創作者類型的文章類別分布")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "creator_content_mix.png"), dpi=150)
    plt.close()
    print("  Saved creator_content_mix.png")


def _plot_frequency(creators_df: pd.DataFrame, plots_dir: str):
    """Scatter: posts/week vs avg likes, colored by type."""
    plt.figure(figsize=(12, 8))

    type_colors = dict(zip(CREATOR_TYPES, sns.color_palette("Set2", len(CREATOR_TYPES))))

    for ctype in CREATOR_TYPES:
        mask = creators_df["primary_type"] == ctype
        subset = creators_df[mask]
        if subset.empty:
            continue
        plt.scatter(
            subset["posts_per_week"],
            subset["avg_likes"],
            label=ctype,
            color=type_colors[ctype],
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5,
        )

    plt.xlabel("每週發文數")
    plt.ylabel("平均按讚數")
    plt.title("發文頻率 vs 互動表現")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "posting_frequency.png"), dpi=150)
    plt.close()
    print("  Saved posting_frequency.png")


def _plot_engagement_by_tier(creators_df: pd.DataFrame, plots_dir: str):
    """Bar chart: engagement by follower tier."""
    df = creators_df.copy()
    df["tier"] = df["followers"].apply(get_follower_tier)

    tier_order = ["0-100", "100-1k", "1k-10k", "10k-100k"]
    df["tier"] = pd.Categorical(df["tier"], categories=tier_order, ordered=True)
    df = df[df["tier"].notna()]

    tier_stats = (
        df.groupby("tier", observed=True)
        .agg(
            creators=("username", "count"),
            avg_likes=("avg_likes", "mean"),
            avg_replies=("avg_replies", "mean"),
        )
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(tier_stats))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], tier_stats["avg_likes"], width, label="平均按讚", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], tier_stats["avg_replies"], width, label="平均回覆", color="#3498db", alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{t}\n({int(c)} 人)" for t, c in zip(tier_stats.index, tier_stats["creators"])])
    ax.set_ylabel("平均數")
    ax.set_title("各粉絲級距的互動表現")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "engagement_by_tier.png"), dpi=150)
    plt.close()
    print("  Saved engagement_by_tier.png")


def _plot_top_creators(top_df: pd.DataFrame, plots_dir: str):
    """Table-style visualization of top creators per type."""
    if top_df.empty:
        return

    types_with_data = top_df["type"].unique()
    n_types = len(types_with_data)
    fig, axes = plt.subplots(n_types, 1, figsize=(14, 2.5 * n_types))
    if n_types == 1:
        axes = [axes]

    for ax, ctype in zip(axes, types_with_data):
        type_data = top_df[top_df["type"] == ctype]
        ax.axis("off")
        ax.set_title(ctype, fontsize=12, fontweight="bold", loc="left")

        table_data = []
        for _, row in type_data.iterrows():
            table_data.append([
                f"#{row['rank']}",
                row["username"],
                f"{row['followers']:,}",
                f"{row['avg_likes']:.1f}",
                f"{row['post_count']}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=["排名", "帳號", "粉絲數", "平均讚", "貼文數"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

    plt.suptitle("各類型 Top 3 創作者", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_creators.png"), dpi=150)
    plt.close()
    print("  Saved top_creators.png")


# ── Main entry point ──────────────────────────────────────────────────────────


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(scripts_dir, "../data")
    article_cache_path = os.path.join(scripts_dir, "outputs/cache/article-classifications.json")
    creator_cache_path = os.path.join(scripts_dir, "outputs/cache/creator-classifications.json")
    output_dir = os.path.join(scripts_dir, "outputs/tables")
    plots_dir = os.path.join(scripts_dir, "outputs/plots/creator-classification")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Phase 1: Load & Feature Extraction ──
    print("=" * 60)
    print("Creator Classification Analysis")
    print("=" * 60)

    print(f"\nLoading data from {data_dir}...")
    df = load_data(data_dir)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    # Ensure followers is numeric
    if "followers" not in df.columns:
        df["followers"] = 0
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)

    # Filter same as other scripts
    df = df[df["followers"] <= 100000].copy()
    df = remove_extreme_outliers(df, column="like_count", threshold=1000)
    df = df[df["text"].str.strip().str.len() > 0].copy()

    print(f"After filtering: {len(df)} posts from {df['username'].nunique()} creators.")

    # Load profiles & article cache
    profiles = load_profiles(data_dir)
    article_cache = load_article_cache(article_cache_path)

    # Build creator features
    creators_df = build_creator_features(df, profiles, article_cache)

    # ── Phase 2: Gemini Classification ──
    print("\n" + "=" * 60)
    print("Phase 2: AI Classification (Gemini)")
    print("=" * 60)

    model = init_gemini()
    creators_df = load_or_classify_creators(creators_df, creator_cache_path, model)

    print("\nClassification Summary:")
    print(creators_df["primary_type"].value_counts().to_string())

    # ── Phase 3: Analysis ──
    print("\n" + "=" * 60)
    print("Phase 3: Analysis")
    print("=" * 60)

    dist_df = analyze_type_distribution(creators_df)
    engagement_df = analyze_type_engagement(creators_df)
    tier_cross = analyze_type_by_tier(creators_df)
    top_df = analyze_top_creators(creators_df)
    mix_df = analyze_content_mix_by_type(creators_df)
    freq_df = analyze_posting_frequency(creators_df)

    # ── Phase 4: Plots ──
    print("\n" + "=" * 60)
    print("Phase 4: Visualization")
    print("=" * 60)

    plot_all(
        creators_df, dist_df, engagement_df, tier_cross,
        top_df, mix_df, freq_df, plots_dir,
    )

    # ── Phase 5: Export ──
    print("\n" + "=" * 60)
    print("Phase 5: Export")
    print("=" * 60)

    # Full creator table
    export_cols = [
        "username", "full_name", "followers", "post_count",
        "avg_likes", "median_likes", "max_likes", "avg_replies", "avg_reposts",
        "posts_per_week", "preferred_hour",
        "primary_type", "secondary_type", "confidence", "reasoning",
        "dominant_article_category",
    ]
    existing_cols = [c for c in export_cols if c in creators_df.columns]
    export_df = creators_df[existing_cols].sort_values("avg_likes", ascending=False)
    export_path = os.path.join(output_dir, "creator-classification.csv")
    export_df.to_csv(export_path, index=False, encoding="utf-8-sig")
    print(f"Saved creator classification to {export_path}")

    # Summary
    summary_path = os.path.join(output_dir, "creator-classification-summary.csv")
    dist_df.to_csv(summary_path, encoding="utf-8-sig")
    print(f"Saved summary to {summary_path}")

    # Engagement
    eng_path = os.path.join(output_dir, "creator-classification-engagement.csv")
    engagement_df.to_csv(eng_path, encoding="utf-8-sig")
    print(f"Saved engagement to {eng_path}")

    # Top creators
    if not top_df.empty:
        top_path = os.path.join(output_dir, "creator-classification-top.csv")
        top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
        print(f"Saved top creators to {top_path}")

    print("\n" + "=" * 60)
    print("Done! All outputs saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
