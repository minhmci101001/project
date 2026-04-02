import os
import pandas as pd
import numpy as np
import emoji
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix
import joblib

def contains_emoji(text):
    return 1 if emoji.emoji_count(str(text)) > 0 else 0

def is_question(text):
    return 1 if '?' in str(text) else 0

def is_exclamation(text):
    return 1 if '!' in str(text) else 0

def has_number_in_text(text):
    import re
    return 1 if bool(re.search(r'\d', str(text))) else 0

def feature_engineering(df):
    """Xử lý và tạo thêm đặc trưng từ dữ liệu gốc."""
    df_model = df.copy()
    
    # 1. Tính toán Tỷ lệ tương tác (Engagement Rate) - ĐÃ LOẠI BỎ (DATA LEAKAGE)
    # LÝ DO: Engagement rate phụ thuộc vào like, comment, và view, những chỉ số chỉ có sau khi phát hành.
    # df_model['engagement_rate'] = (df_model['like_count'] + df_model['comment_count']) / (df_model['view_count'] + 1)

    
    # 2. Xử lý Văn bản Tiêu Đề (Text Analysis)
    df_model['title_length'] = df_model['title'].astype(str).str.len()
    df_model['has_emoji'] = df_model['title'].apply(contains_emoji)
    df_model['is_question'] = df_model['title'].apply(is_question)
    df_model['is_exclamation'] = df_model['title'].apply(is_exclamation)
    df_model['has_number'] = df_model['title'].apply(has_number_in_text)

    # 3. Tags
    df_model['tags_count'] = df_model['tags'].apply(lambda x: len(str(x).split('|')) if pd.notnull(x) else 0)
    
    # 4. Thời gian (Time Analysis)
    df_model['published_at'] = pd.to_datetime(df_model['published_at'])
    df_model['publish_hour'] = df_model['published_at'].dt.hour
    df_model['day_of_week'] = df_model['published_at'].dt.weekday # 0 = Monday, 6 = Sunday
    
    # 5. Đặc trưng Kênh & Danh mục
    df_model['category_id'] = pd.to_numeric(df_model['category_id'], errors='coerce').fillna(24).astype(int)
    # Tránh giá trị rỗng của Subscriber
    if 'subscriber_count' not in df_model.columns:
        df_model['subscriber_count'] = 0
    df_model['subscriber_count'] = pd.to_numeric(df_model['subscriber_count'], errors='coerce').fillna(0).astype(int)
    # Log transform channel size để giảm thiểu outlier
    df_model['log_subscriber_count'] = np.log1p(df_model['subscriber_count'])
    
    # Các feature quyết định (Trước khi đăng video)
    features = [
        'duration_seconds', 
        'title_length', 'has_emoji', 'is_question', 'is_exclamation', 'has_number',
        'tags_count', 'category_id', 'log_subscriber_count', 
        'publish_hour', 'day_of_week'
    ]
    
    X = df_model[features]
    y = df_model['is_trending']
    
    # Xử lý null cuối
    X = X.fillna(0)
    
    return X, y, features

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- BASELINE MODEL (LOGISTIC REGRESSION) ---
    print("\n" + "="*50)
    print("🤖 ĐÁNH GIÁ MÔ HÌNH CƠ SỞ (BASELINE - LOGISTIC REGRESSION)")
    print("="*50)
    
    # Chuẩn hóa dữ liệu cho Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    baseline_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    baseline_model.fit(X_train_scaled, y_train)
    
    # Evaluate Baseline
    y_pred_base = baseline_model.predict(X_test_scaled)
    y_prob_base = baseline_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Baseline Accuracy:  {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"Baseline F1-Score:  {f1_score(y_test, y_pred_base):.4f}")
    print(f"Baseline ROC-AUC:   {roc_auc_score(y_test, y_prob_base):.4f}")
    print("\nClassification Report (Baseline):\n", classification_report(y_test, y_pred_base))
    
    
    # --- PROPOSED MODEL (RANDOM FOREST V2) ---
    print("\n" + "="*50)
    print("🚀 ĐÁNH GIÁ MÔ HÌNH ĐỀ XUẤT (RANDOM FOREST V2)")
    print("="*50)
    
    # Random Forest với xử lý mất cân bằng dữ liệu
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # Đánh giá Random Forest
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    print(f"RF Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"RF F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"RF ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}")
    
    print("\nConfusion Matrix (RF):\n", confusion_matrix(y_test, y_pred_rf))
    print("\nClassification Report (RF):\n", classification_report(y_test, y_pred_rf))
    
    # Phân tích mức độ quan trọng RF
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📌 MỨC ĐỘ QUAN TRỌNG CỦA TỪNG ĐẶC TRƯNG (RF):")
    print(feature_importance)
    
    return rf_model

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_path = os.path.join(project_dir, 'data', 'youtube_data.csv')
    model_dir = os.path.join(project_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(project_dir, 'models', 'trending_rf_model.joblib')
    
    df = pd.read_csv(data_path)
    X, y, features = feature_engineering(df)
    
    model = train_and_evaluate(X, y)
    
    joblib.dump(model, model_path)
    print(f"\n💾 Đã lưu mô hình V2 thành công tại: {model_path}")
