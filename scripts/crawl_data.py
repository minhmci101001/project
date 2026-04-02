import os
import time
import pandas as pd
from googleapiclient.discovery import build
import isodate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv('YOUTUBE_API_KEY')

# Prevent running if API Key is not set
if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
    print("❌ LỖI: Chưa có YouTube API Key!")
    print("👉 Hãy tạo một file tên là .env trong thư mục youtube_trending_project.")
    print("👉 Điền nội dung sau vào file .env:")
    print("YOUTUBE_API_KEY=Mã_API_Của_Bạn")
    exit(1)

# Build the YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_details(video_ids):
    """Lấy chi tiết thống kê và siêu dữ liệu cho một danh sách ID video."""
    all_videos = []
    # YouTube API cho phép tối đa 50 ID trong một lần gọi
    for i in range(0, len(video_ids), 50):
        chunk_ids = video_ids[i:i+50]
        try:
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(chunk_ids)
            )
            response = request.execute()
            
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                content_details = item.get("contentDetails", {})
                
                # Parse duration ISO 8601 to seconds
                duration_iso = content_details.get("duration", "PT0S")
                duration_sec = isodate.parse_duration(duration_iso).total_seconds()
                
                # Trích xuất dữ liệu
                video_data = {
                    "video_id": item["id"],
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "category_id": snippet.get("categoryId", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "tags": "|".join(snippet.get("tags", [])),
                    "duration_seconds": duration_sec,
                    "view_count": int(statistics.get("viewCount", 0)),
                    "like_count": int(statistics.get("likeCount", 0)),
                    "comment_count": int(statistics.get("commentCount", 0))
                }
                all_videos.append(video_data)
        except Exception as e:
            print(f"Lỗi khi lấy chi tiết video: {e}")
            
    # Lấy thông tin lượng Subscriber của các kênh này
    channel_ids = list(set([v["channel_id"] for v in all_videos if v.get("channel_id")]))
    channel_stats = {}
    
    print(f"  👉 Đang lấy thông số (Subscriber Count) cho {len(channel_ids)} kênh YouTube...")
    for i in range(0, len(channel_ids), 50):
        chunk_cids = channel_ids[i:i+50]
        try:
            request = youtube.channels().list(
                part="statistics",
                id=','.join(chunk_cids)
            )
            response = request.execute()
            for item in response.get("items", []):
                channel_stats[item["id"]] = int(item.get("statistics", {}).get("subscriberCount", 0))
        except Exception as e:
            print(f"Lỗi khi lấy chi tiết kênh: {e}")
            
    # Gắn subscriber_count vào từng video
    for v in all_videos:
        v["subscriber_count"] = channel_stats.get(v["channel_id"], 0)
        
    return all_videos

def get_trending_videos(region_code="VN", max_results=300):
    """Lấy danh sách các video đang làm mưa làm gió (Trending)."""
    print(f"🚀 Bắt đầu lấy danh sách video Trending (Quốc gia: {region_code})...")
    trending_videos = []
    next_page_token = None
    
    while len(trending_videos) < max_results:
        try:
            request = youtube.videos().list(
                part="id",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get("items", []):
                trending_videos.append(item["id"])
                
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except Exception as e:
            print(f"Lỗi lấy video trending: {e}")
            break
            
    print(f"✅ Đã tìm thấy {len(trending_videos)} video Trending. Đang tải chi tiết...")
    videos_data = get_video_details(trending_videos)
    
    # Gắn nhãn phân loại (Trending = 1)
    for v in videos_data:
        v["is_trending"] = 1
        
    return videos_data

def get_non_trending_videos(queries=["vlog", "giải trí", "tin tức", "trò chơi", "âm nhạc", "đánh giá"], max_per_query=50):
    """Lấy danh sách các video ngẫu nhiên (Non-Trending) để làm bộ đối chứng."""
    print("🚀 Bắt đầu lấy danh sách video bình thường (Non-Trending)...")
    non_trending_ids = []
    
    for query in queries:
        print(f"  👉 Đang tìm kiếm từ khóa: {query}")
        try:
            # Dùng search để tìm video mới
            request = youtube.search().list(
                part="id",
                q=query,
                type="video",
                order="date", # Lấy video gần đây (chưa hoặc không lọt trending)
                maxResults=max_per_query
            )
            response = request.execute()
            
            for item in response.get("items", []):
                non_trending_ids.append(item["id"]["videoId"])
                
        except Exception as e:
            print(f"Lỗi khi tìm từ khóa '{query}': {e}")
            continue
                
    # Lọc trùng lặp
    non_trending_ids = list(set(non_trending_ids))
    print(f"✅ Đã tìm thấy {len(non_trending_ids)} video Non-Trending. Đang tải chi tiết...")
    
    videos_data = get_video_details(non_trending_ids)
    
    # Gắn nhãn phân loại (Trending = 0)
    for v in videos_data:
        v["is_trending"] = 0
        
    return videos_data

if __name__ == "__main__":
    # Đảm bảo thư mục data tồn tại
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Lấy dữ liệu Trending
    trending_data = get_trending_videos(region_code="VN", max_results=200)
    
    # 2. Lấy dữ liệu Non-Trending
    non_trending_data = get_non_trending_videos(max_per_query=40)
    
    # 3. Gộp lại và tạo DataFrame
    all_data = trending_data + non_trending_data
    df = pd.DataFrame(all_data)
    
    # 4. Lưu ra file CSV
    output_path = os.path.join(data_dir, "youtube_data.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig") # utf-8-sig cho Excel đọc đúng tiếng Việt
    
    print(f"\n🎉 HOÀN THÀNH!")
    print(f"📊 Đã tạo dataset với tổng {len(df)} video.")
    print(f"  - TinTrending: {len(trending_data)}")
    print(f"  - Bình thường: {len(non_trending_data)}")
    print(f"📁 Dữ liệu được lưu tại: {output_path}")
