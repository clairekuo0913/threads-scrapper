import os
import sys
from datetime import timedelta, timezone

import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import load_data, set_chinese_font, remove_extreme_outliers

# Define timezone (UTC+8 for Taiwan)
TZ_TW = timezone(timedelta(hours=8))


def plot_analysis(df: pd.DataFrame, output_dir: str):
    """
    Generate and save plots for the analysis.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_chinese_font()

    # 1. Hourly Engagement
    plt.figure(figsize=(12, 6))
    hourly_stats = df.groupby("hour")[["like_count", "reply_count"]].mean()
    hourly_stats.plot(kind="bar", figsize=(12, 6))
    plt.title("Hourly Average Engagement")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hourly_engagement.png"))
    plt.close()

    # 2. Commercial Slots Engagement
    plt.figure(figsize=(12, 6))
    # Order for plotting
    slot_order = [
        "Morning Commute",
        "Work Morning",
        "Lunch",
        "Work Afternoon",
        "Evening Commute",
        "Prime Time",
        "Late Night",
    ]
    # Filter only existing slots
    existing_slots = [s for s in slot_order if s in df["commercial_slot"].unique()]

    slot_stats = df.groupby("commercial_slot")[["like_count"]].mean()
    # Reindex to ensure logical order
    if not slot_stats.empty:
        slot_stats = slot_stats.reindex(existing_slots)

    slot_stats.plot(kind="bar", figsize=(12, 6), color="orange", legend=False)
    plt.title("Engagement by Commercial Time Slot")
    plt.xlabel("Time Slot")
    plt.ylabel("Average Likes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "commercial_slots_engagement.png"))
    plt.close()

    # 3. Text Length Engagement
    plt.figure(figsize=(12, 6))
    length_stats = df.groupby("length_bin", observed=False)[["like_count"]].mean()
    length_stats.plot(kind="bar", figsize=(12, 6), color="green", legend=False)
    plt.title("Engagement by Text Length")
    plt.xlabel("Text Length")
    plt.ylabel("Average Likes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_engagement.png"))
    plt.close()

    # 4. Sentences Count Engagement
    plt.figure(figsize=(12, 6))
    sentences_stats = df.groupby("sentences_bin", observed=False)[["like_count"]].mean()
    sentences_stats.plot(kind="bar", figsize=(12, 6), color="purple", legend=False)
    plt.title("Engagement by Sentences Count")
    plt.xlabel("Sentences Count")
    plt.ylabel("Average Likes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentences_count_engagement.png"))
    plt.close()

    print(f"Plots saved to {output_dir}")


def analyze_time_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform time-based analysis:
    1. Hourly
    2. Equal Intervals (e.g., 4 hours)
    3. Commercial Time Slots
    """
    print("\n--- Time-Based Analysis ---")

    # 1. Hourly Analysis
    df["hour"] = df["posted_at_tw"].dt.hour
    
    # Enhanced aggregation with distinct creators and median
    hourly_stats = df.groupby("hour").agg({
        "like_count": ["mean", "median", "count"],
        "reply_count": ["mean", "median"],
        "repost_count": ["mean", "median"],
        "username": "nunique"
    })

    # Flatten column names
    hourly_stats.columns = [
        "_".join(col).strip() for col in hourly_stats.columns.values
    ]
    hourly_stats = hourly_stats.rename(columns={"username_nunique": "distinct_creators"})

    print("\n1. Hourly Average Engagement (Top 5 hours by Likes):")
    # Rename for clarity in print output
    display_stats = hourly_stats.sort_values("like_count_mean", ascending=False).head(5)
    print(
        display_stats[
            [
                "like_count_mean",
                "like_count_median",
                "like_count_count",
                "distinct_creators",
            ]
        ]
    )

    # 2. Equal Time Intervals (e.g., 6 blocks of 4 hours)
    # 0-4, 4-8, 8-12, 12-16, 16-20, 20-24
    bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ["Late Night", "Early Morning", "Morning", "Afternoon", "Evening", "Night"]
    df["time_interval"] = pd.cut(
        df["hour"], bins=bins, labels=labels, right=False, include_lowest=True
    )

    interval_stats = df.groupby("time_interval", observed=False).agg({
        "like_count": ["mean", "median", "count"],
        "reply_count": ["mean", "median"],
        "repost_count": ["mean", "median"],
        "username": "nunique"
    })

    # Flatten column names
    interval_stats.columns = [
        "_".join(col).strip() for col in interval_stats.columns.values
    ]
    interval_stats = interval_stats.rename(columns={"username_nunique": "distinct_creators"})

    print("\n2. Equal Time Intervals Analysis:")
    print(
        interval_stats[
            [
                "like_count_mean",
                "like_count_median",
                "like_count_count",
                "distinct_creators",
            ]
        ]
    )

    # 3. Commercial Time Slots
    # Morning Commute: 07:00 - 09:00
    # Work Morning: 09:00 - 12:00
    # Lunch: 12:00 - 13:30
    # Work Afternoon: 13:30 - 18:00
    # Evening Commute: 18:00 - 19:30
    # Prime Time: 19:30 - 23:00
    # Late Night: 23:00 - 07:00

    def get_commercial_slot(dt):
        h = dt.hour
        m = dt.minute
        t = h + m / 60.0

        if 7 <= t < 9:
            return "Morning Commute"
        elif 9 <= t < 12:
            return "Work Morning"
        elif 12 <= t < 13.5:
            return "Lunch"
        elif 13.5 <= t < 18:
            return "Work Afternoon"
        elif 18 <= t < 19.5:
            return "Evening Commute"
        elif 19.5 <= t < 23:
            return "Prime Time"
        else:
            return "Late Night"

    df["commercial_slot"] = df["posted_at_tw"].apply(get_commercial_slot)

    slot_stats = df.groupby("commercial_slot").agg({
        "like_count": ["mean", "median", "count"],
        "reply_count": ["mean", "median"],
        "repost_count": ["mean", "median"],
        "username": "nunique"
    })

    # Flatten column names
    slot_stats.columns = ["_".join(col).strip() for col in slot_stats.columns.values]
    slot_stats = slot_stats.rename(columns={"username_nunique": "distinct_creators"})

    # Let's sort by like_count for "effectiveness"
    print("\n3. Commercial Time Slots Analysis (Sorted by Likes):")
    print(
        slot_stats.sort_values("like_count_mean", ascending=False)[
            [
                "like_count_mean",
                "like_count_median",
                "like_count_count",
                "distinct_creators",
            ]
        ]
    )

    return df


def analyze_content_form(df):
    """
    Analyze content form and layout:
    1. Text length bins
    2. Sentences count bins
    """
    print("\n--- Content Form Analysis ---")

    # Calculate metrics
    df["text_length"] = df["text"].astype(str).str.len()
    df["sentences_count"] = (
        df["text"]
        .astype(str)
        .apply(lambda x: len([p for p in x.split("\n") if p.strip()]))
    )

    # 1. Text Length Bins
    # Create bins: 0-50, 50-100, 100-200, 200-500, 500+
    length_bins = [0, 50, 100, 200, 500, 1000, 5000]
    length_labels = [
        "Short (<50)",
        "Medium-Short (50-100)",
        "Medium (100-200)",
        "Medium-Long (200-500)",
        "Long (500-1000)",
        "Very Long (1000+)",
    ]

    df["length_bin"] = pd.cut(
        df["text_length"], bins=length_bins, labels=length_labels, right=False
    )

    length_stats = df.groupby("length_bin", observed=False).agg({
        "like_count": ["mean", "median", "count"],
        "reply_count": ["mean", "median"],
        "repost_count": ["mean", "median"],
        "username": "nunique"
    })

    # Flatten column names
    length_stats.columns = [
        "_".join(col).strip() for col in length_stats.columns.values
    ]
    length_stats = length_stats.rename(columns={"username_nunique": "distinct_creators"})

    print("\n1. Text Length 'Sweet Spot' (Average Engagement):")
    print(
        length_stats[
            [
                "like_count_mean",
                "like_count_median",
                "like_count_count",
                "distinct_creators",
            ]
        ]
    )

    # 2. Sentences Count Bins
    # With right=False, bins are [a, b), so [0,1) captures 0, [1,2) captures 1, etc.
    sentences_bins = [1, 2, 4, 7, 11, 15, 21, 26, 31, 36, 41, 46, 100]
    sentences_labels = [
        "1",  # [1, 2) → 1
        "2-3",  # [2, 4) → 2, 3
        "4-6",  # [4, 7) → 4, 5, 6
        "7-10",  # [7, 11) → 7, 8, 9, 10
        "11-14",  # [11, 15) → 11, 12, 13, 14
        "15-20",  # [15, 21) → 15-20
        "21-25",  # [21, 26) → 21-25
        "26-30",  # [26, 31) → 26-30
        "31-35",  # [31, 36) → 31-35
        "36-40",  # [36, 41) → 36-40
        "41-45",  # [41, 46) → 41-45
        "46+",  # [46, 100) → 46+
    ]

    df["sentences_bin"] = pd.cut(
        df["sentences_count"], bins=sentences_bins, labels=sentences_labels, right=False
    )

    sentences_stats = df.groupby("sentences_bin", observed=False).agg({
        "like_count": ["mean", "median", "count"],
        "reply_count": ["mean", "median"],
        "repost_count": ["mean", "median"],
        "username": "nunique"
    })

    # Flatten column names
    sentences_stats.columns = [
        "_".join(col).strip() for col in sentences_stats.columns.values
    ]
    sentences_stats = sentences_stats.rename(columns={"username_nunique": "distinct_creators"})

    print("\n2. Sentences Count 'Sweet Spot' (Average Engagement):")
    print(
        sentences_stats[
            [
                "like_count_mean",
                "like_count_median",
                "like_count_count",
                "distinct_creators",
            ]
        ]
    )

    return df


def main():
    # Path relative to this script (scripts/basic/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(script_dir)  # scripts/
    data_dir = os.path.join(scripts_dir, "../data")

    print(f"Loading data from {data_dir}...")
    df = load_data(data_dir)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(df)} posts.")

    # Filter out accounts with too many followers (>100k)
    if "followers" in df.columns:
        df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)
        original_count = len(df)
        original_users = df["username"].nunique()
        
        # Filter out accounts with > 100k followers
        df = df[df["followers"] <= 100000]
        
        filtered_count = len(df)
        filtered_users = df["username"].nunique()
        removed_posts = original_count - filtered_count
        removed_users = original_users - filtered_users
        
        print(f"\nFiltered out {removed_posts} posts from {removed_users} account(s) with >100k followers.")
        print(f"Remaining: {filtered_count} posts from {filtered_users} accounts.")

    # Remove extreme outliers (viral posts with > 1000 likes)
    df = remove_extreme_outliers(df, column="like_count", threshold=1000)

    # Run analyses
    df = analyze_time_based(df)
    df = analyze_content_form(df)

    # Save processed data to outputs/
    output_dir = os.path.join(scripts_dir, "outputs/tables")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "basic-analysis.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved analyzed data to {output_path}")

    # Generate plots to outputs/
    plots_dir = os.path.join(scripts_dir, "outputs/plots/basic-analysis")
    plot_analysis(df, plots_dir)


if __name__ == "__main__":
    main()
