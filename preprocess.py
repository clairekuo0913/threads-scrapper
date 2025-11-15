from bs4 import BeautifulSoup


def preprocess_threads_html(
    input_file="threads_page_source.html", output_file="threads_cleaned.html"
):
    """
    讀取原始的 Threads HTML 檔案，移除不必要的標籤和屬性，並儲存為清理後的版本。
    """
    print(f"正在讀取檔案: {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        print(
            f"錯誤: 找不到檔案 '{input_file}'。請確認檔案存在於腳本執行的相同目錄下。"
        )
        return

    print("開始使用 BeautifulSoup 解析 HTML...")
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. 移除所有不需要的標籤
    # 這些標籤通常不包含貼文的主要文字內容
    print("正在移除 <script>, <style>, <link>, <meta>, <svg> 標籤...")
    tags_to_remove = ["script", "style", "link", "meta", "svg"]
    for tag in soup.find_all(tags_to_remove):
        tag.decompose()  # .decompose() 會將標籤及其所有內容從解析樹中完全移除

    # 根據您的要求，也可以移除所有圖片標籤
    # print("正在移除 <img> 標籤...")
    # for tag in soup.find_all('img'):
    #     tag.decompose()

    # 2. 移除所有標籤中不需要的屬性，例如 inline style, class 等
    print("正在移除所有標籤中的 style, class, 及 data-* 屬性...")
    # soup(True) 是一個快捷方式，可以找到檔案中的所有標籤
    for tag in soup(True):
        # 使用一個 list 來收集所有要刪除的屬性，避免在迭代過程中修改字典
        attrs_to_delete = []
        for attr in tag.attrs:
            # 如果屬性是 'style', 'class', 或以 'data-' 開頭，就加入待刪除列表
            if attr == "style" or attr == "class" or attr.startswith("data-"):
                attrs_to_delete.append(attr)

        # 實際執行刪除
        for attr in attrs_to_delete:
            del tag[attr]

    # 3. 取得清理後的 HTML，並使用 .prettify() 進行格式化
    print("正在格式化清理後的 HTML...")
    cleaned_html = soup.prettify()

    # 4. 將清理後的 HTML 寫入新檔案
    print(f"正在儲存清理後的 HTML 至 {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_html)

    print("\n預處理完成！")
    print(f"清理後的檔案已儲存為: {output_file}")


if __name__ == "__main__":
    preprocess_threads_html()
