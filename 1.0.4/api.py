from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import json
import logging

app = Flask(__name__)

# Load model, scaler, and feature names
with open('final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('fitted_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

def create_input_data(year, month):
    num_days = pd.Period(f'{year}-{month}').days_in_month
    date_range = pd.date_range(start=f'{year}-{month}-01', periods=num_days, freq='D')

    categories = ['Clothing', 'Coffe', 'Communal', 'Events', 'Film/enjoyment', 'Fuel', 'Health',
                  'Learning', 'Market', 'Motel', 'Other', 'Phone', 'Rent Car', 'Restuarant',
                  'Sport', 'Taxi', 'Tech', 'Transport', 'Travel', 'joy']

    dfs = []
    for category in categories:
        df = pd.DataFrame({
            'year': year,
            'month': month,
            'hours': np.repeat(12, num_days),
            'weekday': date_range.dayofweek + 1,
            'day_of_year': date_range.dayofyear,
            'category': np.repeat(category, num_days)
        })

        df = pd.get_dummies(df, columns=['category'], prefix='category', drop_first=False)
        expected_categories = ['category_' + cat for cat in categories]
        for exp_cat in expected_categories:
            if exp_cat not in df.columns:
                df[exp_cat] = 0

        df = df[['hours', 'weekday', 'year', 'day_of_year', 'month'] + expected_categories]
        dfs.append(df)

    return pd.concat(dfs)

@app.route('/predict', methods=['GET'])
def predict_expenses():
    date_str = request.args.get('date', default=datetime.now().strftime('%m.%Y'), type=str)
    try:
        month, year = map(int, date_str.split('.'))
        df_input = create_input_data(year, month)
        
        logging.info(f'Input Data: {df_input.head()}')
        
        df_scaled = scaler.transform(df_input)
        
        logging.info(f'Scaled Input Data: {df_scaled[:5]}')
        
        predictions = model.predict(df_scaled)
        
        logging.info(f'Raw Predictions: {predictions[:5]}')

        category_expenses = {}
        total_expenses = 0
        category_prefix = "category_"
        for col in df_input.columns:
            if col.startswith(category_prefix):
                category_name = col[len(category_prefix):]
                total = predictions[df_input[col] == 1].sum()
                if total > 0:
                    rounded_total = round(total, 2)
                    category_expenses[category_name] = rounded_total
                    total_expenses += rounded_total

        result = {
            "category_expenses": category_expenses,
            "total_predicted_expense": round(total_expenses, 2)
        }
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
