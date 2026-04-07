import sqlite3
import os

def query_crowd_data(db_path="tracking_results/crowd_detection.db"):
    """
    查詢人流檢測資料庫中的所有記錄
    """
    if not os.path.exists(db_path):
        print(f"資料庫文件不存在: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查詢所有記錄
    cursor.execute("SELECT video_name, people_count, processed_at FROM video_stats ORDER BY processed_at DESC")
    rows = cursor.fetchall()

    print("人流檢測資料庫記錄:")
    print("=" * 60)
    for row in rows:
        video_name, people_count, processed_at = row
        print(f"視頻: {video_name}")
        print(f"人數: {people_count} 人")
        print(f"處理時間: {processed_at}")
        print("-" * 40)

    conn.close()

if __name__ == "__main__":
    query_crowd_data()