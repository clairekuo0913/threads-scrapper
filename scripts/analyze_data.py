import glob
import json
import os
import platform
from datetime import timedelta, timezone

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

# Define timezone (UTC+8 for Taiwan)
TZ_TW = timezone(timedelta(hours=8))


def set_chinese_font():
    """
    Attempt to set a Chinese-compatible font for matplotlib.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        # Common Chinese fonts on macOS
        # Try to find a font that actually exists
        candidates = ["Arial Unicode MS", "PingFang TC", "Heiti TC", "STHeiti"]
        for cand in candidates:
            try:
                # check if font exists
                if len(fm.findfont(cand, fallback_to_default=False)) > 0:
                    plt.rcParams["font.family"] = cand
                    return
            except Exception as e:
                print(f"Error setting font: {e}")
                continue
    elif system == "Windows":
        plt.rcParams["font.family"] = ["Microsoft JhengHei", "SimHei"]
        return
    elif system == "Linux":
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback"]
        return

    # Fallback
    print(
        "Warning: Could not set a specific Chinese font. Plots may show boxes for Chinese characters."
    )


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


def load_data(data_dir):
    """
    Load all JSON files from the data directory into a pandas DataFrame.
    """
    all_posts = []

    # Walk through the data directory
    # Structure: data/<username>/<timestamp>.json
    pattern = os.path.join(data_dir, "*", "*.json")
    files = glob.glob(pattern)

    print(f"Found {len(files)} JSON files.")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            profile = data.get("profile", {})
            username = profile.get("username", "unknown")
            posts = data.get("posts", [])

            for post in posts:
                post["username"] = username
                if "text" not in post:
                    post["text"] = ""
                all_posts.append(post)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not all_posts:
        print("No posts found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)

    # Handle different timestamp fields
    if "posted_at" not in df.columns:
        df["posted_at"] = pd.NaT

    if "published_on" in df.columns:
        # Fill missing posted_at with published_on
        # Convert unix timestamp to UTC datetime
        mask = df["posted_at"].isna() & df["published_on"].notna()
        # pd.to_datetime with unit='s' returns naive datetime (UTC usually).
        # We localize to UTC.
        df.loc[mask, "posted_at"] = pd.to_datetime(
            df.loc[mask, "published_on"], unit="s", utc=True
        )

    # Convert posted_at to datetime with utc=True to handle mixed formats/timezones uniformly
    # Use errors='coerce' to handle bad data
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True, errors="coerce")

    # Drop rows with invalid timestamps
    if df["posted_at"].isna().any():
        print(
            f"Warning: {df['posted_at'].isna().sum()} rows with invalid timestamps dropped."
        )
        df = df.dropna(subset=["posted_at"])

    # Now df['posted_at'] is definitely datetime64[ns, UTC]
    df["posted_at_tw"] = df["posted_at"].dt.tz_convert(TZ_TW)

    return df


def analyze_time_based(df):
    """
    Perform time-based analysis:
    1. Hourly
    2. Equal Intervals (e.g., 4 hours)
    3. Commercial Time Slots
    """
    print("\n--- Time-Based Analysis ---")

    # 1. Hourly Analysis
    df["hour"] = df["posted_at_tw"].dt.hour
    hourly_stats = df.groupby("hour")[
        ["like_count", "reply_count", "repost_count"]
    ].mean()
    print("\n1. Hourly Average Engagement (Top 5 hours by Likes):")
    print(hourly_stats.sort_values("like_count", ascending=False).head(5))

    # 2. Equal Time Intervals (e.g., 6 blocks of 4 hours)
    # 0-4, 4-8, 8-12, 12-16, 16-20, 20-24
    bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ["Late Night", "Early Morning", "Morning", "Afternoon", "Evening", "Night"]
    df["time_interval"] = pd.cut(
        df["hour"], bins=bins, labels=labels, right=False, include_lowest=True
    )

    interval_stats = df.groupby("time_interval", observed=False)[
        ["like_count", "reply_count", "repost_count"]
    ].mean()
    print("\n2. Equal Time Intervals Analysis:")
    print(interval_stats)

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

    slot_stats = df.groupby("commercial_slot")[
        ["like_count", "reply_count", "repost_count"]
    ].mean()
    # Sort by a logical order if needed, but pandas sort is alphabetical by default or index
    # Let's sort by like_count for "effectiveness"
    print("\n3. Commercial Time Slots Analysis (Sorted by Likes):")
    print(slot_stats.sort_values("like_count", ascending=False))

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

    length_stats = df.groupby("length_bin", observed=False)[
        ["like_count", "reply_count", "repost_count"]
    ].mean()
    print("\n1. Text Length 'Sweet Spot' (Average Engagement):")
    print(length_stats)

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

    sentences_stats = df.groupby("sentences_bin", observed=False)[
        ["like_count", "reply_count", "repost_count"]
    ].mean()
    print("\n2. Sentences Count 'Sweet Spot' (Average Engagement):")
    print(sentences_stats)

    return df


def main():
    # Path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")

    print(f"Loading data from {data_dir}...")
    df = load_data(data_dir)

    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(df)} posts.")

    # Run analyses
    df = analyze_time_based(df)
    df = analyze_content_form(df)

    # Optional: Save processed data
    output_path = os.path.join(script_dir, "analyzed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved analyzed data to {output_path}")

    # Generate plots
    plots_dir = os.path.join(script_dir, "plots")
    plot_analysis(df, plots_dir)


if __name__ == "__main__":
    main()
