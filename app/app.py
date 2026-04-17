import os
import re
import numpy as np
import pandas as pd
import joblib
import emoji as emoji_lib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Resolve paths relative to this file so the app works from any working directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path  = os.path.join(project_dir, 'models', 'trending_rf_model.joblib')

model = None
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
    else:
        print(f"ERROR: Model file not found at: {model_path}")
        print("Please run scripts/model_training.py first.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please run scripts/model_training.py first.'}), 500

    try:
        data = request.json

        # Extract user-provided inputs
        title            = str(data.get('title', ''))
        duration_seconds = float(data.get('duration_seconds', 0))
        tags_count       = float(data.get('tags_count', 0))
        category_id      = int(data.get('category_id', 24))
        subscriber_count = float(data.get('subscriber_count', 0))
        publish_hour     = float(data.get('publish_hour', 12))
        day_of_week      = int(data.get('day_of_week', 0))

        # Optional UI overrides (checkbox signals sent from the frontend)
        ui_has_emoji   = data.get('ui_has_emoji',   None)
        ui_is_question = data.get('ui_is_question', None)
        ui_has_number  = data.get('ui_has_number',  None)

        # NLP pre-processing
        title_length = len(title)

        # Priority logic: use UI override when provided, otherwise analyse the title text
        has_emoji   = int(ui_has_emoji)   if ui_has_emoji   is not None else (1 if emoji_lib.emoji_count(title) > 0 else 0)
        is_question = int(ui_is_question) if ui_is_question is not None else (1 if '?' in title else 0)
        has_number  = int(ui_has_number)  if ui_has_number  is not None else (1 if bool(re.search(r'\d', title)) else 0)
        is_exclamation = 1 if '!' in title else 0

        # Numeric pre-processing
        log_subscriber_count = np.log1p(subscriber_count)

        # Build feature DataFrame in the exact column order the model expects
        features = pd.DataFrame([{
            'duration_seconds':     duration_seconds,
            'title_length':         title_length,
            'has_emoji':            has_emoji,
            'is_question':          is_question,
            'is_exclamation':       is_exclamation,
            'has_number':           has_number,
            'tags_count':           tags_count,
            'category_id':          category_id,
            'log_subscriber_count': log_subscriber_count,
            'publish_hour':         publish_hour,
            'day_of_week':          day_of_week
        }])

        features = features[[
            'duration_seconds',
            'title_length', 'has_emoji', 'is_question', 'is_exclamation', 'has_number',
            'tags_count', 'category_id', 'log_subscriber_count',
            'publish_hour', 'day_of_week'
        ]]

        # Predict
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # P(class=1 | Trending)

        return jsonify({
            'prediction':  int(prediction),
            'probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\nStarting YouTube Trending Predictor — Web Demo Server...")
    app.run(debug=True, port=8000)
