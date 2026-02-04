import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import load_data, set_chinese_font


def calculate_gini(array):
    """Calculate the of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from:
    # http://www.statsdirect.com/help/def∆ault.htm#nonparametric_methods/gini.htm
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array.astype(float)
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def plot_distribution(data, title, xlabel, output_path, log_scale=False, bins=30):
    plt.figure(figsize=(10, 6))
    if log_scale:
        # Filter non-positive values for log scale
        plot_data = data[data > 0]
        sns.histplot(plot_data, bins=bins, kde=True, log_scale=True)
        plt.title(f"{title} (Log Scale)")
    else:
        sns.histplot(data, bins=bins, kde=True)
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_boxplot(data, title, ylabel, output_path, showfliers=False):
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data, showfliers=showfliers)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_dataset():
    # Setup paths (scripts/dataset/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(script_dir)  # scripts/
    data_dir = os.path.join(scripts_dir, "../data")
    output_dir = os.path.join(scripts_dir, "outputs/tables")
    plots_dir = os.path.join(scripts_dir, "outputs/plots/dataset-analysis")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    set_chinese_font()

    print("Loading data...")
    df = load_data(data_dir)

    if df.empty:
        print("No data found.")
        return

    # User-level aggregation
    if "followers" in df.columns:
        df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)

    user_stats = (
        df.groupby("username")
        .agg(
            {
                "followers": "first",  # Assuming followers count is relatively static or taking the latest
                "text": "count",
                "like_count": "sum",
                "reply_count": "sum",
                "repost_count": "sum",
            }
        )
        .rename(columns={"text": "post_count"})
    )

    # Identify large account (>100k) for reporting purpose
    large_accounts = user_stats[user_stats["followers"] > 100000]
    print(f"Found {len(large_accounts)} accounts with > 100k followers.")

    # We will analyze the filtered dataset as per user's preference in previous tasks
    # But for dataset analysis, it's good to show what we have.
    # Let's produce stats for the "Active/Target Dataset" (excluding the outlier if exists)

    # Filter for the main analysis set (<= 100k followers)
    valid_users = user_stats[user_stats["followers"] <= 100000].index
    df_filtered = df[df["username"].isin(valid_users)].copy()
    user_stats_filtered = user_stats[user_stats.index.isin(valid_users)].copy()

    # --- Metrics Calculation ---

    stats_summary = []

    def get_distribution_metrics(series, name, unit=""):
        desc = series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        gini = calculate_gini(series.values)
        cv = series.std() / series.mean() if series.mean() != 0 else 0

        return {
            "Metric": name,
            "Unit": unit,
            "Count": int(desc["count"]),
            "Mean": round(desc["mean"], 2),
            "Median": round(desc["50%"], 2),
            "Std": round(desc["std"], 2),
            "Min": desc["min"],
            "Max": desc["max"],
            "P25": desc["25%"],
            "P75": desc["75%"],
            "P90": desc["90%"],
            "P99": desc["99%"],
            "CV": round(cv, 2),  # Coefficient of Variation (Dispersal)
            "Gini": round(gini, 3),  # Inequality
        }

    # 1. User Level Metrics
    stats_summary.append(
        get_distribution_metrics(
            user_stats_filtered["followers"], "User Followers", "users"
        )
    )
    stats_summary.append(
        get_distribution_metrics(
            user_stats_filtered["post_count"], "Posts per User", "posts"
        )
    )

    # 2. Post Level Metrics
    df_filtered["text_len"] = df_filtered["text"].astype(str).str.len()
    stats_summary.append(
        get_distribution_metrics(df_filtered["text_len"], "Post Text Length", "chars")
    )
    stats_summary.append(
        get_distribution_metrics(df_filtered["like_count"], "Likes per Post", "likes")
    )
    stats_summary.append(
        get_distribution_metrics(
            df_filtered["reply_count"], "Replies per Post", "replies"
        )
    )
    stats_summary.append(
        get_distribution_metrics(
            df_filtered["repost_count"], "Reposts per Post", "reposts"
        )
    )

    # Create Summary DataFrame
    summary_df = pd.DataFrame(stats_summary)
    summary_path = os.path.join(output_dir, "dataset_stats_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n--- Dataset Statistics Summary ---")
    print(summary_df.to_string())

    # --- Visualizations ---

    # 1. User Distributions
    plot_distribution(
        user_stats_filtered["followers"],
        "Distribution of User Followers",
        "Followers",
        os.path.join(plots_dir, "dist_user_followers.png"),
        log_scale=True,
    )

    plot_distribution(
        user_stats_filtered["post_count"],
        "Distribution of Posts per User",
        "Number of Posts",
        os.path.join(plots_dir, "dist_user_posts.png"),
    )

    # 2. Post Content Distributions
    plot_distribution(
        df_filtered["text_len"],
        "Distribution of Text Length",
        "Character Count",
        os.path.join(plots_dir, "dist_text_length.png"),
    )

    # 3. Engagement Boxplots (Log scale often better for engagement or just excluding outliers)
    # We use showfliers=False to see the box clearly, and maybe another one with outliers?
    # Let's just do standard boxplots first, clipping extreme outliers for visualization if needed

    plot_boxplot(
        df_filtered["like_count"],
        "Likes Distribution (No Outliers)",
        "Likes",
        os.path.join(plots_dir, "box_likes.png"),
        showfliers=False,
    )

    plot_boxplot(
        df_filtered["reply_count"],
        "Replies Distribution (No Outliers)",
        "Replies",
        os.path.join(plots_dir, "box_replies.png"),
        showfliers=False,
    )

    # 4. Correlation Matrix
    corr_cols = ["text_len", "like_count", "reply_count", "repost_count"]
    corr = df_filtered[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Post Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"))
    plt.close()

    print(f"\nAnalysis complete. Plots saved to {plots_dir}")


if __name__ == "__main__":
    analyze_dataset()
