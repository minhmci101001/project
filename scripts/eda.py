import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Thiết lập phong cách xịn xò cho biểu đồ
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="pastel")
# Thiết lập font chữ tiếng Việt (tuỳ chọn)
try:
    plt.rcParams['font.family'] = 'Segoe UI'
except:
    pass

def load_data(filepath):
    print(f"Đang tải dữ liệu từ {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Đã tải xong tập dữ liệu với {df.shape[0]} dòng và {df.shape[1]} cột.")
    return df

def feature_engineering_eda(df):
    """Tính toán thêm các đặc trưng phục vụ trực quan hóa."""
    df_plot = df.copy()
    
    # Độ dài tiêu đề
    df_plot['title_length'] = df_plot['title'].astype(str).apply(len)
    
    # Tính Engagement Rate (Tỷ lệ tương tác)
    # Thêm 1 vào mẫu số để tránh chia cho 0
    df_plot['engagement_rate'] = (df_plot['like_count'] + df_plot['comment_count']) / (df_plot['view_count'] + 1)
    
    # Label Text
    df_plot['Trending Status'] = df_plot['is_trending'].map({1: 'Trending', 0: 'Bình thường'})
    
    # Parser thời gian
    if 'published_at' in df_plot.columns:
        df_plot['published_at'] = pd.to_datetime(df_plot['published_at'])
        df_plot['publish_hour'] = df_plot['published_at'].dt.hour
    else:
        df_plot['publish_hour'] = 12 # Backup fallback
    
    return df_plot


def run_eda(df_plot, output_dir):
    print("\n🎨 Bắt đầu vẽ biểu đồ chuyên sâu...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Biểu đồ 1: Tỷ lệ phân bổ Class 
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df_plot, x='Trending Status', palette=['#3498db', '#e74c3c'])
    plt.title("Phân bổ Tập dữ liệu (Trending vs Cân bằng)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Số lượng video", fontsize=12)
    plt.xlabel("")
    for i in ax.containers: ax.bar_label(i, padding=3, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=300)
    plt.close()

    # 2. Biểu đồ 2: Phân phối lượt Xem (View Count) - Trục Log
    plt.figure(figsize=(10, 6))
    df_plot['log_views'] = np.log1p(df_plot['view_count'])
    sns.kdeplot(data=df_plot, x='log_views', hue='Trending Status', fill=True, palette=['#3498db', '#e74c3c'], common_norm=False, alpha=0.5)
    plt.title("Phân phối Lượt Xem (Log Scale) theo Trending Status", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Log1p(View Count)", fontsize=12)
    plt.ylabel("Mật độ", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_view_distribution.png"), dpi=300)
    plt.close()

    # 3. Biểu đồ 3: Correlation Matrix của các chỉ số Engagement
    plt.figure(figsize=(10, 8))
    numeric_cols = ['is_trending', 'view_count', 'like_count', 'comment_count', 'duration_seconds', 'title_length', 'engagement_rate']
    corr = df_plot[numeric_cols].corr()
    # Tạo mask để che nửa trên tốn diện tích
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                linewidths=1, cbar_kws={"shrink": .8})
    plt.title("Ma trận tương quan (Correlation Heatmap)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_correlation_matrix.png"), dpi=300)
    plt.close()

    # 4. Biểu đồ 4: Độ dài Tiêu đề có ảnh hưởng tới xu hướng không?
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_plot, x='Trending Status', y='title_length', palette=['#3498db', '#e74c3c'], showfliers=False)
    plt.title("Độ dài Tiêu đề vs Khả năng lên Trending", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Độ dài Tiêu đề (ký tự)", fontsize=12)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_title_length_boxplot.png"), dpi=300)
    plt.close()
    
    # 5. Biểu đồ 5: Engagement Rate
    plt.figure(figsize=(8, 6))
    # Loại trừ các điểm hỏng quá lớn để visualize đẹp hơn
    sns.boxplot(data=df_plot, x='Trending Status', y='engagement_rate', palette=['#3498db', '#e74c3c'], showfliers=False)
    plt.title("Tỷ lệ Tương tác (Like+Cmt/View) ở 2 nhóm", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Engagement Rate", fontsize=12)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_engagement_rate_boxplot.png"), dpi=300)
    plt.close()
    
    # 6. Biểu đồ 6: Phân phối theo Danh mục (Category)
    plt.figure(figsize=(12, 6))
    category_map = {10: 'Music', 20: 'Gaming', 22: 'People/Blogs', 23: 'Comedy', 24: 'Entertainment', 25: 'News', 27: 'Education', 28: 'Tech'}
    df_plot['Category Name'] = df_plot['category_id'].map(category_map).fillna('Other')
    ax2 = sns.countplot(data=df_plot, x='Category Name', hue='Trending Status', palette=['#3498db', '#e74c3c'])
    plt.title("Phân phối danh mục nội dung (Category Distribution)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Số lượng video", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_category_distribution.png"), dpi=300)
    plt.close()

    # 7. Biểu đồ 7: Phân phối theo Giờ đăng bài (Publish Hour)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_plot, x='publish_hour', hue='Trending Status', multiple="dodge", bins=24, palette=['#3498db', '#e74c3c'], shrink=0.8)
    plt.title("Tác động của Giờ đăng bài đến khả năng lên Trending", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Số lượng video", fontsize=12)
    plt.xlabel("Giờ trong ngày (0-23)", fontsize=12)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_publish_hour_distribution.png"), dpi=300)
    plt.close()

    print(f"🎉 Đã tạo thành công 7 biểu đồ phân tích sâu! Các file ảnh được lưu tại: {output_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'youtube_data.csv')
    output_dir = os.path.join(os.path.dirname(current_dir), 'plots')
    
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {data_path}")
        print("👉 Vui lòng chạy file crawl_data.py trước!")
    else:
        df = load_data(data_path)
        df_engineered = feature_engineering_eda(df)
        run_eda(df_engineered, output_dir)
