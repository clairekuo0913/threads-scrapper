import os

import pandas as pd
from utils import load_data


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


def calculate_tier_stats():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")

    print("Loading data...")
    df = load_data(data_dir)

    if df.empty:
        print("No data found.")
        return

    # Ensure numeric
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    df["reply_count"] = pd.to_numeric(df["reply_count"], errors="coerce").fillna(0)

    # Add tier
    df["tier"] = df["followers"].apply(get_follower_tier)

    # Calculate engagement
    df["total_engagement"] = df["like_count"] + df["reply_count"]

    # Calculate Engagement Rate per post: (Like + Reply) / Followers
    # Note: If followers is 0, ER is 0 to avoid division by zero
    df["engagement_rate"] = df.apply(
        lambda x: (x["total_engagement"] / x["followers"]) if x["followers"] > 0 else 0,
        axis=1,
    )

    # Group by tier
    tier_stats = (
        df.groupby("tier")
        .agg(
            {
                "username": "nunique",  # User count
                "id": "count",  # Post count
                "total_engagement": "sum",  # Total interactions
                "engagement_rate": "mean",  # Average ER per post
                "followers": "mean",  # Average followers
            }
        )
        .rename(
            columns={
                "username": "user_count",
                "id": "post_count",
                "total_engagement": "total_interactions",
                "engagement_rate": "avg_engagement_rate",
            }
        )
    )

    # Reorder index
    tier_order = ["0-100", "100-1k", "1k-10k", "10k-100k", "100k-1m", "1m+"]
    tier_stats = tier_stats.reindex([t for t in tier_order if t in tier_stats.index])

    print("\n--- Tier Overview ---")
    print(tier_stats.to_markdown())


if __name__ == "__main__":
    calculate_tier_stats()
