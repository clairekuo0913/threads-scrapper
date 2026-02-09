---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Threads social media scraping and analysis platform that collects user profiles and posts for data analysis. The project consists of two main phases: web scraping using Selenium and data analysis using pandas/matplotlib.

## Architecture

### Core Components

- **Web Scrapers**: Selenium-based scrapers that collect Threads user data
  - `fetch_users.py` - Searches for users by keywords and builds a user list
  - `fetch_posts.py` - Scrapes individual user profiles and their posts

- **Analysis Pipeline**: Modular data analysis scripts in the `scripts/` directory
  - `scripts/shared/utils.py` - Common utilities for data loading, text cleaning, and plotting
  - `scripts/dataset/analysis.py` - Dataset distribution analysis (Gini coefficients, follower distributions)
  - `scripts/basic/analysis.py` - Basic engagement metrics and patterns
  - `scripts/word/analysis.py` - Text analysis including word frequency and categorization

### Data Flow

1. **User Discovery**: `fetch_users.py` searches by keywords → saves to `data/discovered_users.json`
2. **Profile Scraping**: `fetch_posts.py` reads user list → scrapes profiles and posts → saves JSON files to `data/<username>/`
3. **Analysis**: Analysis scripts load all JSON data → generate insights and visualizations → save to `scripts/outputs/`

### Data Structure

Scraped data follows this schema (defined in `fetch_posts.py` using Pydantic models):
- **Profile**: username, full_name, followers, bio, bio_links
- **ThreadPost**: id, text, url, posted_at, like_count, reply_count, repost_count, forward_count

## Development Commands

### Environment Setup
```bash
# Install dependencies with UV (recommended)
uv sync

# Or with pip
pip install -e .
```

### Running Scrapers
```bash
# First-time login and cookie setup
python fetch_posts.py

# Search for users by keywords (defined in script)
python fetch_users.py

# Scrape specific user
python fetch_posts.py <username>
# or
python fetch_posts.py https://www.threads.net/@username

# Batch scrape all users from discovered_users.json
python fetch_posts.py
```

### Running Analysis
```bash
# Dataset distribution analysis
python scripts/dataset/analysis.py

# Basic engagement metrics
python scripts/basic/analysis.py

# Word/text analysis
python scripts/word/analysis.py
```

## Key Technical Details

### Authentication
- Uses cookie-based authentication stored in `threads_cookies.pkl`
- First run of `fetch_posts.py` prompts for manual login
- Cookies are automatically reused in subsequent runs

### Rate Limiting and Scraping Strategy
- Headless Chrome with specific user agent configuration
- Built-in delays between requests (10 seconds default)
- Incremental scrolling to load dynamic content
- Deduplication of posts using post IDs

### Text Processing
- Chinese text support with font configuration for matplotlib
- Text cleaning removes Instagram metadata and noise patterns ("已靜音", "更多\n")
- Taiwan timezone handling (UTC+8) for temporal analysis

### Analysis Architecture
- Modular analysis scripts that can run independently
- Shared utilities for consistent data loading and visualization
- Outputs saved to `scripts/outputs/` with structured naming
- Support for follower-based segmentation and engagement analysis

## File Locations

- **Scraped Data**: `data/<username>/<timestamp>.json`
- **User Lists**: `data/discovered_users.json`
- **Analysis Outputs**: `scripts/outputs/plots/` and `scripts/outputs/tables/`
- **Documentation**: `docs/` (GitBook format with analysis reports)

## Important Notes

- The `data/` directory is gitignored and contains sensitive user data
- ChromeDriver must be installed and accessible (macOS ARM64 version included)
- Cookie file (`threads_cookies.pkl`) contains login credentials and should not be committed
- All scripts handle Chinese text and require appropriate font configuration for visualizations
