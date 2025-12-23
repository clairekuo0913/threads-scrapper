import json
import os
import pickle
import sys
import time
import urllib.parse
from typing import Set

from bs4 import BeautifulSoup
from selenium import webdriver

# Constants
COOKIES_FILE = "threads_cookies.pkl"
THREADS_BASE_URL = "https://www.threads.net"
DATA_DIR = "data"
MAX_SEARCH_SCROLLS = 5
SCROLL_PAUSE_TIME = 5


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    # options.add_argument("--headless") # Comment out headless for debugging if needed, or keep it. User didn't specify. Keeping it consistent with fetch_posts might be safer, but search usually requires more "human-like" behavior. Let's keep it headless for now as per previous script.
    options.add_argument("--headless")
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=options)
    return driver


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
        print("找不到Cookies檔案。請先執行 fetch_posts.py 進行登入。")
        return False
    except Exception as e:
        print(f"載入 Cookies 時發生錯誤: {e}")
        return False


def ensure_login(driver):
    if not load_cookies(driver, COOKIES_FILE):
        print("無法登入，請檢查 cookies 檔案。")
        sys.exit(1)
    driver.refresh()
    time.sleep(3)


def fetch_users_by_keyword(driver, keyword: str) -> Set[str]:
    encoded_keyword = urllib.parse.quote(keyword)
    url = (
        f"https://www.threads.net/search?q={encoded_keyword}&serp_type=default&hl=zh-tw"
    )
    print(f"正在搜尋: {keyword} ({url})")

    driver.get(url)
    time.sleep(5)  # Wait for search results to load

    usernames = set()

    # Scroll and scrape
    last_height = driver.execute_script("return document.body.scrollHeight")
    for scroll in range(MAX_SEARCH_SCROLLS):
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Look for links to user profiles
        # Typically these are <a href="/@username">
        # We want to avoid posts which might also link to users, but usually search results prioritize users or mix them.
        # The search page for Threads usually has a specific structure.
        # Let's grab all hrefs starting with /@ and filter out post links if they look different
        # or just grab the username.

        links = soup.find_all("a", href=True)
        for link in links:
            href = link.get("href")
            # Match /@username but NOT /@username/post/...
            if href.startswith("/@") and "/post/" not in href:
                username = href.replace("/@", "").strip("/")
                if username:
                    usernames.add(username)

        print(
            f"滾動 {scroll + 1}/{MAX_SEARCH_SCROLLS}, 目前找到 {len(usernames)} 個使用者"
        )

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    return usernames


def main():
    keywords = ["創業", "行銷", "ㄧ人公司"]
    output_file = "discovered_users.json"
    all_users = set()

    # Load existing users if file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_users = json.load(f)
                all_users.update(existing_users)
            print(f"已讀取現有 {len(all_users)} 個使用者")
        except Exception as e:
            print(f"讀取現有檔案時發生錯誤: {e}")

    driver = setup_driver()
    try:
        ensure_login(driver)

        for keyword in keywords:
            users = fetch_users_by_keyword(driver, keyword)
            print(f"關鍵字 '{keyword}' 找到 {len(users)} 個使用者")
            all_users.update(users)

        print(f"\n總共累積 {len(all_users)} 個不重複使用者")

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(list(all_users), f, ensure_ascii=False, indent=2)

        print(f"使用者列表已更新至 {output_file}")

    except Exception as e:
        print(f"發生錯誤: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
