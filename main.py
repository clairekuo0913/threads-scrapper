import pickle
import time

import requests
import selenium
from selenium import webdriver

THREADS_URL = "https://www.threads.com/@danielwchen0"
COOKIES_FILE = "threads_cookies.pkl"
DOWNLOAD_FOLDER = "threads_images"
SCROLL_PAUSE_TIME = 2  # 每次滾動後等待新內容載入的時間
MAX_SCROLLS = 10  # 最大滾動次數，避免無限滾動
OUTPUT_HTML_FILE = "threads_page_source.html"


def _version_check():
    print("version check: ")
    print(f"requests: {requests.__version__}")
    print(f"selenium: {selenium.__version__}")


def setup_driver():
    """設定並返回一個Chrome WebDriver實例"""
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # 如果不需要瀏覽器介面，可以啟用無頭模式
    # options.add_argument("--headless")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    return driver


def save_cookies(driver, location):
    """手動登入後儲存Cookies"""
    print("請手動登入Threads，完成後在終端機按下Enter鍵...")
    input()
    pickle.dump(driver.get_cookies(), open(location, "wb"))
    print("Cookies 已成功儲存！")


def load_cookies(driver, location):
    """載入Cookies以自動登入"""
    try:
        cookies = pickle.load(open(location, "rb"))
        # 需要先訪問一次網站的根網域才能設定cookie
        driver.get(THREADS_URL)
        time.sleep(1)  # 等待頁面基礎架構載入
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


def scroll_page(driver):
    """動態滾動頁面以載入更多貼文"""
    print("開始動態滾動頁面...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0

    while scroll_count < MAX_SCROLLS:
        # 滾動到頁面底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 等待新內容載入
        time.sleep(SCROLL_PAUSE_TIME)

        # 計算新的頁面高度，如果與上次相同，表示已到底部
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("已到達頁面底部。")
            break
        last_height = new_height
        scroll_count += 1
        print(f"已滾動 {scroll_count} 次...")
    print("滾動結束。")


def save_html_source(driver, file_path):
    """獲取當前頁面的完整HTML原始碼並儲存"""
    print(f"正在將頁面HTML原始碼儲存至 {file_path} ...")
    try:
        # driver.page_source 會回傳當前瀏覽器渲染的完整HTML
        html_content = driver.page_source
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("HTML原始碼儲存成功！")
    except Exception as e:
        print(f"儲存HTML時發生錯誤: {e}")


def main():
    driver = setup_driver()
    driver.get(THREADS_URL)

    if not load_cookies(driver, COOKIES_FILE):
        save_cookies(driver, COOKIES_FILE)

    driver.refresh()
    print("頁面已重新整理，等待5秒讓內容載入...")
    time.sleep(5)

    scroll_page(driver)

    save_html_source(driver, OUTPUT_HTML_FILE)

    driver.quit()
    print(f"任務完成！請查看 {OUTPUT_HTML_FILE} 檔案。")


if __name__ == "__main__":
    main()
