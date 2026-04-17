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
    print("ERROR: YouTube API Key is missing!")
    print("Please create a .env file in the youtube_trending_project directory.")
    print("Add the following content to the .env file:")
    print("YOUTUBE_API_KEY=Your_API_Key_Here")
    exit(1)

# Build the YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_details(video_ids):
    """Fetch statistical details and metadata for a list of video IDs."""
    all_videos = []
    # YouTube API allows a maximum of 50 IDs per request
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
                
                # Extract data
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
            print(f"Error fetching video details: {e}")
            
    # Get Subscriber counts for these channels
    channel_ids = list(set([v["channel_id"] for v in all_videos if v.get("channel_id")]))
    channel_stats = {}
    
    print(f"  👉 Fetching Subscriber Count for {len(channel_ids)} YouTube channels...")
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
            print(f"Error fetching channel details: {e}")
            
    # Attach subscriber_count to each video
    for v in all_videos:
        v["subscriber_count"] = channel_stats.get(v["channel_id"], 0)
        
    return all_videos

def get_trending_videos(region_code="VN", max_results=300):
    """Fetch a list of currently trending videos."""
    print(f"Fetching Trending videos list (Region: {region_code})...")
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
            print(f"Error fetching trending videos: {e}")
            break
            
    print(f"Found {len(trending_videos)} Trending videos. Fetching details...")
    videos_data = get_video_details(trending_videos)
    
    # Assign classification label (Trending = 1)
    for v in videos_data:
        v["is_trending"] = 1
        
    return videos_data

def get_non_trending_videos(queries=["vlog", "entertainment", "news", "gaming", "music", "reviews"], max_per_query=50):
    """Fetch a list of random (Non-Trending) videos for the control group."""
    print("Fetching Non-Trending videos list...")
    non_trending_ids = []
    
    for query in queries:
        print(f"Searching for keyword: {query}")
        try:
            # Use search to find new videos
            request = youtube.search().list(
                part="id",
                q=query,
                type="video",
                order="date", # Get recent videos (that haven't/won't trend)
                maxResults=max_per_query
            )
            response = request.execute()
            
            for item in response.get("items", []):
                non_trending_ids.append(item["id"]["videoId"])
                
        except Exception as e:
            print(f"Error searching for keyword '{query}': {e}")
            continue
                
    # Remove duplicates
    non_trending_ids = list(set(non_trending_ids))
    print(f"Found {len(non_trending_ids)} Non-Trending videos. Fetching details...")
    
    videos_data = get_video_details(non_trending_ids)
    
    # Assign classification label (Trending = 0)
    for v in videos_data:
        v["is_trending"] = 0
        
    return videos_data

if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Fetch Trending data
    trending_data = get_trending_videos(region_code="VN", max_results=200)
    
    # 2. Fetch Non-Trending data
    non_trending_data = get_non_trending_videos(max_per_query=40)
    
    # 3. Combine and create DataFrame
    all_data = trending_data + non_trending_data
    df = pd.DataFrame(all_data)
    
    # 4. Save to CSV file
    output_path = os.path.join(data_dir, "youtube_data.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig") # utf-8-sig to display properly in Excel
    
    print(f"COMPLETED!")
    print(f"Created dataset with total {len(df)} videos.")
    print(f"  - Trending: {len(trending_data)}")
    print(f"  - Non-Trending: {len(non_trending_data)}")
    print(f"Data saved to: {output_path}")
