# YouTube Trending Prediction AI

Dự án Machine Learning dự đoán xác suất lọt vào danh sách "Thịnh hành" (Trending) của một video YouTube trước khi được đăng tải, sử dụng thuật toán Random Forest và Logistic Regression Baseline.

## ✨ Tính năng nổi bật
* **Data Crawling**: Tự động thu thập dữ liệu bằng YouTube Data API v3.
* **Exploratory Data Analysis (EDA)**: Phân tích và sinh 7 biểu đồ phân phối dữ liệu (Tỷ lệ tương tác, ảnh hưởng của danh mục, khung giờ vàng đăng bài, v.v.).
* **AI NLP Models**: Áp dụng Random Forest V2 với trọng số cân bằng lớp (balanced weights), đánh giá trực tiếp với Baseline Logistic Regression. Loại bỏ hoàn toàn Data Leakage để mô hình có tính thực tiễn cao nhất.
* **Modern Web Interface**: Giao diện Flask kết hợp phong cách Glassmorphism đầy tương tác (Interactive Title Features).

## 🚀 Khởi chạy dự án (Local)

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Khởi chạy Web Server:
```bash
python app/app.py
```
3. Mở trình duyệt và truy cập `http://127.0.0.1:8000/`.

## 📁 Cấu trúc thư mục
- `app/`: Giao diện web chạy bằng Flask (HTML/CSS/JS).
- `data/`: Dữ liệu gốc thu thập từ API YouTube.
- `models/`: Chứa file Joblib nén của mô hình AI Random Forest V2.
- `plots/`: Các biểu đồ kết xuất thông qua phân tích mô tả.
- `scripts/`: Chứa kịch bản thu thập API (`crawl_data.py`), vẽ biểu đồ (`eda.py`), và huấn luyện mô hình (`model_training.py`).
