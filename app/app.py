import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Khởi tạo đường dẫn để tải mô hình
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_dir, 'models', 'trending_rf_model.joblib')

model = None
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Đã tải mô hình thành công từ: {model_path}")
    else:
        print(f"❌ Không tìm thấy file mô hình tại: {model_path}")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Mô hình chưa được tải. Hãy đảm bảo bạn đã chạy scripts/model_training.py'}), 500
        
    try:
        data = request.json
        
        # Trích xuất dữ liệu do người dùng nhập vào
        title = str(data.get('title', ''))
        duration_seconds = float(data.get('duration_seconds', 0))
        tags_count = float(data.get('tags_count', 0))
        category_id = int(data.get('category_id', 24))
        subscriber_count = float(data.get('subscriber_count', 0))
        publish_hour = float(data.get('publish_hour', 12))
        day_of_week = int(data.get('day_of_week', 0))
        
        # Lấy giá trị override từ UI (nếu có)
        ui_has_emoji = data.get('ui_has_emoji', None)
        ui_is_question = data.get('ui_is_question', None)
        ui_has_number = data.get('ui_has_number', None)
        
        # Tiền xử lý NLP
        import emoji
        import re
        title_length = len(title)
        
        # Logic ưu tiên: Nếu UI gửi true/false, dùng giá trị đó. Nếu không (None), tự phân tích từ text.
        has_emoji = int(ui_has_emoji) if ui_has_emoji is not None else (1 if emoji.emoji_count(title) > 0 else 0)
        is_question = int(ui_is_question) if ui_is_question is not None else (1 if '?' in title else 0)
        has_number = int(ui_has_number) if ui_has_number is not None else (1 if bool(re.search(r'\d', title)) else 0)
        
        is_exclamation = 1 if '!' in title else 0
        
        # Tiền xử lý số
        import numpy as np
        log_subscriber_count = np.log1p(subscriber_count)
        
        # Dataframe
        features = pd.DataFrame([{
            'duration_seconds': duration_seconds,
            'title_length': title_length,
            'has_emoji': has_emoji,
            'is_question': is_question,
            'is_exclamation': is_exclamation,
            'has_number': has_number,
            'tags_count': tags_count,
            'category_id': category_id,
            'log_subscriber_count': log_subscriber_count,
            'publish_hour': publish_hour,
            'day_of_week': day_of_week
        }])
        
        # Sắp xếp đúng thứ tự cột Model cần
        features = features[[
            'duration_seconds', 
            'title_length', 'has_emoji', 'is_question', 'is_exclamation', 'has_number',
            'tags_count', 'category_id', 'log_subscriber_count', 
            'publish_hour', 'day_of_week'
        ]]
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # Xác suất thuộc class 1 (Trending)
        
        # Trả về kết quả
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n🌐 Bắt đầu khởi động Web Server dành cho Demo Trình bày...")
    app.run(debug=True, port=8000)
    
