import datetime
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from selenium import webdriver

# Constants for scraping
SCROLL_PAUSE_TIME = 10
MAX_SCROLLS = 10
COOKIES_FILE = "threads_cookies.pkl"
THREADS_BASE_URL = "https://www.threads.net"
DATA_DIR = "data"


class Profile(BaseModel):
    username: str
    full_name: Optional[str] = None
    followers: Optional[int] = None
    bio: Optional[str] = None
    bio_links: List[str] = Field(default_factory=list)


class ThreadPost(BaseModel):
    id: str
    text: str
    url: str
    posted_at: Optional[str] = None
    like_count: int
    reply_count: int
    repost_count: int
    forward_count: Optional[int] = None


class ThreadsSnapshot(BaseModel):
    profile: Profile
    posts: List[ThreadPost]


def decode_shortcode_to_id(shortcode):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    media_id = 0
    for char in shortcode:
        media_id = (media_id * 64) + alphabet.index(char)
    return str(media_id)


def clean_post_text(text):
    """
    Clean post text by removing common noise patterns.

    Removes:
    - "更多\n" prefix
    - "\n讚" and content after it
    - "已靜音" and everything after it (Instagram metadata)
    """
    if "更多\n" in text:
        text = text.split("更多\n", 1)[-1]

    if "\n讚" in text:
        text = text.rsplit("\n讚", 1)[0]

    # Remove "已靜音" and everything after it
    # This removes Instagram logo, username and other metadata
    if "已靜音" in text:
        text = text.split("已靜音")[0]

    return text.strip()


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--headless")
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=options)
    return driver


def save_cookies(driver, location):
    print("請手動登入Threads，完成後在終端機按下Enter鍵...")
    input()
    pickle.dump(driver.get_cookies(), open(location, "wb"))
    print("Cookies 已成功儲存！")


def load_cookies(driver, location):
    try:
        cookies = pickle.load(open(location, "rb"))
        driver.get(THREADS_BASE_URL)
        time.sleep(1)
        for cookie in cookies:
            driver.add_cookie(cookie)
        print("Cookies 已成功載入！")
        return True
    except FileNotFoundError:
        print("找不到Cookies檔案。腳本將會引導您手動登入以建立新的Cookies檔案。")
        return False
    except Exception as e:
        print(f"載入 Cookies 時發生錯誤: {e}")
        return False


def ensure_login(driver):
    if not load_cookies(driver, COOKIES_FILE):
        driver.get(THREADS_BASE_URL)
        save_cookies(driver, COOKIES_FILE)
    driver.refresh()
    print("頁面已重新整理，等待載入...")
    time.sleep(3)


def extract_profile_from_html(soup):
    profile_data = {
        "username": "",
        "full_name": None,
        "followers": 0,
        "bio": None,
        "bio_links": [],
    }
    try:
        # Try to extract bio from meta og:description
        meta_desc = soup.find("meta", property="og:description")
        if meta_desc:
            content = meta_desc.get("content", "")
            if content:
                # Extract followers from meta description first (often more reliable format)
                # "1.4 萬位粉絲 • 0 則串文 • ..."
                # Handle non-breaking spaces (\xa0) which might appear as normal spaces or &nbsp;
                content_clean = content.replace("\xa0", " ").replace("&nbsp;", " ")

                # Regex for followers in meta
                # Matches: "1.4 萬", "1000", "1.5M", "10k" followed by "位粉絲"
                # We need to be careful with spaces
                follower_match = re.search(
                    r"([\d.,]+(?:\s*[萬kKwW])?)\s*位粉絲", content_clean
                )
                if follower_match:
                    num_str = follower_match.group(1).replace(",", "").replace(" ", "")
                    if "萬" in num_str or "w" in num_str.lower():
                        profile_data["followers"] = int(
                            float(num_str.replace("萬", "").replace("w", "")) * 10000
                        )
                    elif "k" in num_str.lower():
                        profile_data["followers"] = int(
                            float(num_str.lower().replace("k", "")) * 1000
                        )
                    elif "m" in num_str.lower():
                        profile_data["followers"] = int(
                            float(num_str.lower().replace("m", "")) * 1000000
                        )
                    else:
                        try:
                            profile_data["followers"] = int(float(num_str))
                        except ValueError:
                            pass

                parts = content.split(" • ")
                if len(parts) >= 3:
                    bio_text = " • ".join(parts[2:])
                    suffix_pattern = r"\s*查看\s+@[^\s]+\s+參與的最新對話。$"
                    bio_text = re.sub(suffix_pattern, "", bio_text, flags=re.MULTILINE)
                    profile_data["bio"] = bio_text.strip()
                elif len(parts) == 2:
                    profile_data["bio"] = ""

        h1s = soup.find_all("h1")
        if h1s:
            profile_data["username"] = h1s[0].get_text(strip=True)
            if len(h1s) > 1:
                profile_data["full_name"] = h1s[1].get_text(strip=True)

        # Fallback extraction from HTML elements if meta extraction failed or we want to double check
        # But usually meta is good enough for the count.
        # However, sometimes HTML has the exact number (e.g. title="13608")

        followers_elem = soup.find(string=re.compile(r"位粉絲"))
        if followers_elem:
            parent = followers_elem.parent
            # Check for exact count in title attribute of inner span: <span title="13,608">1.3 萬</span>位粉絲
            exact_span = parent.find("span", title=True)
            if exact_span:
                try:
                    exact_num = exact_span["title"].replace(",", "")
                    profile_data["followers"] = int(exact_num)
                except ValueError:
                    pass
            elif profile_data["followers"] == 0:
                # Fallback to parsing text if meta failed and no title attribute
                followers_text = followers_elem.parent.get_text(strip=True)
                # Normalize spaces
                followers_text = followers_text.replace("\xa0", " ")
                match = re.search(r"([\d.,]+(?:\s*[萬kKwW])?)\s*位粉絲", followers_text)
                if match:
                    num_str = match.group(1).replace(",", "").replace(" ", "")
                    if "萬" in num_str or "w" in num_str.lower():
                        profile_data["followers"] = int(
                            float(num_str.replace("萬", "").replace("w", "")) * 10000
                        )
                    elif "k" in num_str.lower():
                        profile_data["followers"] = int(
                            float(num_str.lower().replace("k", "")) * 1000
                        )
                    else:
                        try:
                            profile_data["followers"] = int(float(num_str))
                        except ValueError:
                            pass

        if h1s:
            container = h1s[0].parent
            for _ in range(5):
                if (
                    container
                    and followers_elem
                    and followers_elem in container.descendants
                ):
                    break
                if container:
                    container = container.parent

            if container:
                links = container.find_all("a", href=True)
                for link in links:
                    href = link.get("href")
                    if not href.startswith("/") and not href.startswith(
                        "https://www.threads.net"
                    ):
                        profile_data["bio_links"].append(href)

                if profile_data["bio"] is None:
                    container_text = container.get_text(separator="\n", strip=True)
                    known_texts = [
                        profile_data["username"],
                        profile_data["full_name"],
                        "位粉絲",
                        "追蹤中",
                        "提及",
                    ]
                    if followers_elem:
                        known_texts.append(followers_elem.parent.get_text(strip=True))

                    lines = container_text.split("\n")
                    bio_lines = []
                    for line in lines:
                        line = line.strip()
                        is_known = False
                        for k in known_texts:
                            if k and k in line:
                                is_known = True
                                break
                        if not is_known and len(line) > 1:
                            if line.lower() != "instagram":
                                bio_lines.append(line)
                    profile_data["bio"] = "\n".join(bio_lines)
    except Exception as e:
        print(f"Error extracting profile: {e}")
    return Profile(**profile_data)


def extract_posts_from_html(soup) -> List[ThreadPost]:
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, "html.parser")
    posts = []
    like_titles = soup.find_all("title", string="讚")

    for like_title in like_titles:
        try:
            like_svg = like_title.parent
            action_bar = like_svg.find_parent(
                lambda tag: tag.name == "div" and tag.find("title", string="回覆")
            )
            if not action_bar:
                continue

            post_container = action_bar.find_parent(
                lambda tag: tag.name == "div"
                and tag.find("a", href=re.compile(r"/post/"))
            )
            if not post_container:
                post_container = action_bar.parent.parent

            def get_stat_count(title_name):
                title_node = action_bar.find("title", string=title_name)
                if not title_node:
                    return 0
                icon_wrapper = title_node.find_parent(attrs={"role": "button"})
                if not icon_wrapper:
                    icon_wrapper = title_node.parent.parent
                text = icon_wrapper.get_text(strip=True)
                text = text.replace(title_name, "")
                match = re.search(r"(\d+(?:,\d+)*(?:\.\d+)?[kKmM萬]?)", text)
                if match:
                    num_str = match.group(1).replace(",", "")
                    if "萬" in text or "w" in text.lower():
                        return int(
                            float(num_str.replace("萬", "").replace("w", "")) * 10000
                        )
                    if "k" in num_str.lower():
                        return int(float(num_str.lower().replace("k", "")) * 1000)
                    return int(float(num_str))
                return 0

            like_count = get_stat_count("讚")
            reply_count = get_stat_count("回覆")
            repost_count = get_stat_count("轉發")
            forward_count = get_stat_count("分享")

            url = ""
            post_id = ""
            posted_at = None
            url_node = post_container.find("a", href=re.compile(r"/post/"))
            if url_node:
                href = url_node.get("href")
                url = f"https://www.threads.net{href}" if href.startswith("/") else href
                match = re.search(r"/post/([^/?]+)", url)
                if match:
                    shortcode = match.group(1)
                    post_id = decode_shortcode_to_id(shortcode)

                time_node = url_node.find("time")
                if time_node and time_node.has_attr("datetime"):
                    posted_at = time_node.get("datetime")

            raw_text = post_container.get_text(separator="\n", strip=True)
            cleaned_text = clean_post_text(raw_text)

            post_data = ThreadPost(
                id=post_id,
                text=cleaned_text,
                url=url,
                posted_at=posted_at,
                like_count=like_count,
                reply_count=reply_count,
                repost_count=repost_count,
                forward_count=forward_count,
            )
            posts.append(post_data)
        except Exception:
            continue
    return posts


def scrape_profile_data(user_name: str) -> Optional[ThreadsSnapshot]:
    """
    爬取指定 URL 的個人檔案與貼文 (支援虛擬滾動)
    """
    driver = setup_driver()
    try:
        print(f"正在前往: {f'https://www.threads.net/{user_name}'}")
        ensure_login(driver)
        driver.get(f"https://www.threads.net/{user_name}")
        time.sleep(5)  # Wait for initial load

        # 1. Extract Profile Immediately (before scrolling)
        initial_soup = BeautifulSoup(driver.page_source, "html.parser")
        profile = extract_profile_from_html(initial_soup)
        print(f"已提取 Profile: {profile.username}")

        # 2. Incremental Scroll and Extract
        all_posts: List[ThreadPost] = []
        seen_ids: Set[str] = set()

        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count = 0

        print("開始動態滾動與提取...")

        while scroll_count < MAX_SCROLLS:
            # Extract from current view
            current_soup = BeautifulSoup(driver.page_source, "html.parser")
            current_posts = extract_posts_from_html(current_soup)

            new_count = 0
            for p in current_posts:
                if p.id and p.id not in seen_ids:
                    seen_ids.add(p.id)
                    all_posts.append(p)
                    new_count += 1

            print(
                f"已滾動 {scroll_count + 1}/{MAX_SCROLLS} (本次新增 {new_count} 篇, 總計 {len(all_posts)} 篇)"
            )

            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("已到達頁面底部或未載入新內容。")
                break
            last_height = new_height
            scroll_count += 1

        print("滾動結束。")
        return ThreadsSnapshot(profile=profile, posts=all_posts)

    except Exception as e:
        print(f"爬取失敗: {e}")
        return None
    finally:
        driver.quit()


def snapshot_to_json(snapshot: ThreadsSnapshot) -> str:
    dump_method = getattr(snapshot, "model_dump", None)
    data = dump_method() if dump_method else snapshot.dict()
    return json.dumps(data, ensure_ascii=False, indent=2)


def save_data(username: str, snapshot: ThreadsSnapshot):
    user_dir = Path(DATA_DIR) / username
    user_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = user_dir / f"{timestamp}.json"
    file_path.write_text(snapshot_to_json(snapshot), encoding="utf-8")
    print(f"資料已儲存至: {file_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_arg = sys.argv[1]
        # Clean up input if it's a URL
        if "threads.net/" in target_arg:
            target_arg = target_arg.split("threads.net/")[-1].strip("/")

        snapshot = scrape_profile_data(user_name=target_arg)
        if snapshot:
            username = snapshot.profile.username or target_arg
            save_data(username, snapshot)
            print(f"\n成功提取 Profile ({username}) 與 {len(snapshot.posts)} 篇貼文。")
    else:
        # Batch mode from discovered_users.json
        json_path = "data/discovered_users.json"
        if Path(json_path).exists():
            print(f"發現 {json_path}，開始批次爬取...")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    users = json.load(f)

                print(f"共有 {len(users)} 個使用者待處理")
                for i, user in enumerate(users):
                    user_data_dir = Path(DATA_DIR) / user
                    if user_data_dir.exists():
                        print(
                            f"\n[{i + 1}/{len(users)}] 使用者 {user} 資料已存在，跳過。"
                        )
                        continue

                    print(f"\n[{i + 1}/{len(users)}] 處理使用者: {user}")
                    try:
                        snapshot = scrape_profile_data(user)
                        if snapshot:
                            saved_username = snapshot.profile.username or user
                            save_data(saved_username, snapshot)
                            print(f"成功處理: {user}")
                        else:
                            print(f"未能提取資料: {user}")
                    except Exception as e:
                        print(f"處理 {user} 時發生錯誤: {e}")

                    # Sleep to avoid rate limiting
                    if i < len(users) - 1:
                        print("等待 10 秒後繼續...")
                        time.sleep(10)
            except Exception as e:
                print(f"讀取使用者列表失敗: {e}")
        else:
            print("請提供 Username 或確保 discovered_users.json 存在")
            print("用法: python fetch_posts.py [username]")
