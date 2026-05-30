from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = 'xgboost_demand_model.pkl'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print('Model loaded ✅')
    else:
        print(f'Model not found at {MODEL_PATH}')

load_model()

# Exact features from Week 2
FEATURE_COLUMNS = [
    'week', 'week_in_year', 'month', 'quarter', 'is_weekend', 'is_holiday',
    'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured',
    'center_id', 'meal_id', 'op_area',
    'lag_1', 'lag_2', 'lag_4', 'lag_8',
    'rolling_mean_4', 'rolling_mean_8', 'rolling_std_4'
]

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.json

        week             = int(data.get('week', 100))
        checkout_price   = float(data.get('checkout_price', 150))
        base_price       = float(data.get('base_price', 160))
        avg_orders       = float(data.get('avg_recent_orders', 300))

        week_in_year = (week % 52) + 1
        month        = ((week_in_year - 1) // 4) + 1
        quarter      = ((month - 1) // 3) + 1
        is_weekend   = 0
        is_holiday   = 0

        feature_row = {
            'week'                  : week,
            'week_in_year'          : week_in_year,
            'month'                 : month,
            'quarter'               : quarter,
            'is_weekend'            : is_weekend,
            'is_holiday'            : is_holiday,
            'checkout_price'        : checkout_price,
            'base_price'            : base_price,
            'emailer_for_promotion' : int(data.get('emailer_for_promotion', 0)),
            'homepage_featured'     : int(data.get('homepage_featured', 0)),
            'center_id'             : int(data.get('center_id', 55)),
            'meal_id'               : int(data.get('meal_id', 1885)),
            'op_area'               : float(data.get('op_area', 4.0)),
            'lag_1'                 : avg_orders,
            'lag_2'                 : avg_orders,
            'lag_4'                 : avg_orders,
            'lag_8'                 : avg_orders,
            'rolling_mean_4'        : avg_orders,
            'rolling_mean_8'        : avg_orders,
            'rolling_std_4'         : avg_orders * 0.2,
        }

        X = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
        prediction = float(np.clip(model.predict(X)[0], 0, None))

        if prediction < 100:
            stock_level    = 'Low Stock'
            recommendation = 'Order minimal inventory — low demand expected.'
            color          = 'low'
        elif prediction < 400:
            stock_level    = 'Medium Stock'
            recommendation = 'Order moderate inventory — average demand expected.'
            color          = 'medium'
        else:
            stock_level    = 'High Stock'
            recommendation = 'Order high inventory — strong demand expected!'
            color          = 'high'

        discount_pct = round((base_price - checkout_price) / base_price * 100, 1) if base_price > 0 else 0

        return jsonify({
            'predicted_orders': round(prediction),
            'stock_level'     : stock_level,
            'recommendation'  : recommendation,
            'color'           : color,
            'discount_pct'    : discount_pct,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)