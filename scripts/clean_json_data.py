"""
Clean all JSON data files by removing '已靜音' and content after it.

This script reads all JSON files in the data directory, cleans the text field
of each post, and writes the cleaned data back to the same file.
"""

import glob
import json
import os


def clean_text(text: str) -> str:
    """
    Clean post text by removing '已靜音' and everything after it.
    """
    if not text:
        return ""
    
    # Remove '已靜音' and everything after it
    if "已靜音" in text:
        text = text.split("已靜音")[0]
    
    # Strip whitespace
    return text.strip()


def clean_json_file(file_path: str) -> dict:
    """
    Clean a single JSON file and return statistics.
    
    Returns:
        dict: Statistics about the cleaning operation
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        posts = data.get("posts", [])
        cleaned_count = 0
        total_posts = len(posts)
        
        # Clean each post
        for post in posts:
            if "text" in post and post["text"]:
                original_text = post["text"]
                cleaned_text = clean_text(original_text)
                
                if original_text != cleaned_text:
                    post["text"] = cleaned_text
                    cleaned_count += 1
        
        # Write back to file if any changes were made
        if cleaned_count > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {
            "file": file_path,
            "total_posts": total_posts,
            "cleaned_posts": cleaned_count,
            "success": True,
        }
    
    except Exception as e:
        return {
            "file": file_path,
            "total_posts": 0,
            "cleaned_posts": 0,
            "success": False,
            "error": str(e),
        }


def clean_all_json_files(data_dir: str) -> None:
    """
    Clean all JSON files in the data directory.
    """
    # Find all JSON files
    pattern = os.path.join(data_dir, "*", "*.json")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} JSON files to clean.")
    print("=" * 60)
    
    total_files = 0
    total_posts = 0
    total_cleaned = 0
    failed_files = []
    
    for file_path in files:
        result = clean_json_file(file_path)
        
        if result["success"]:
            total_files += 1
            total_posts += result["total_posts"]
            total_cleaned += result["cleaned_posts"]
            
            if result["cleaned_posts"] > 0:
                print(f"✓ {os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}")
                print(f"  Cleaned {result['cleaned_posts']}/{result['total_posts']} posts")
        else:
            failed_files.append(file_path)
            print(f"✗ {file_path}")
            print(f"  Error: {result['error']}")
    
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total posts: {total_posts}")
    print(f"  Posts cleaned: {total_cleaned}")
    
    if failed_files:
        print(f"\n⚠️  Failed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")
    else:
        print(f"\n✅ All files processed successfully!")


if __name__ == "__main__":
    # Path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data")
    
    print(f"Cleaning JSON files in: {data_dir}\n")
    clean_all_json_files(data_dir)
