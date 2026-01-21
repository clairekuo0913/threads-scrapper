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


def set_chinese_font() -> None:
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


def clean_text(text: str) -> str:
    """
    Clean post text by removing '已靜音' and everything after it.
    Also removes other common noise patterns.
    """
    if not text:
        return ""

    # Remove '已靜音' and everything after it
    if "已靜音" in text:
        text = text.split("已靜音")[0]

    # Strip whitespace
    return text.strip()


def load_data(data_dir: str) -> pd.DataFrame:
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
            followers = profile.get("followers", 0)
            posts = data.get("posts", [])

            for post in posts:
                post["username"] = username
                post["followers"] = followers
                if "text" not in post:
                    post["text"] = ""
                else:
                    # Clean the text
                    post["text"] = clean_text(post["text"])
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
