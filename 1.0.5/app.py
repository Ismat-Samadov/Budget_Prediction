from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import calendar
from flask_cors import CORS
import os  # Needed for port environment variable

# Load the saved model, scalers, and feature columns
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
scaler_target = joblib.load('scaler_target.pkl')
feature_columns = joblib.load('feature_columns.pkl')

app = Flask(__name__, static_folder='static') 
CORS(app)  # Enable CORS for cross-origin requests

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Renders the index.html file from templates/

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    year = data['year']
    month = data['month']

    num_days = calendar.monthrange(year, month)[1]

    predictions = {}
    total_monthly_expense = 0

    # Dummy-encoded category columns
    categories = [col for col in feature_columns if 'category_' in col]

    for category in categories:
        category_total = 0

        for day in range(1, num_days + 1):
            day_of_week = calendar.weekday(year, month, day)

            # Construct the input data for prediction
            input_data = pd.DataFrame([[year, month, day, day_of_week]], columns=['year', 'month', 'day', 'day_of_week'])

            # Create dummy category columns
            category_data = pd.DataFrame(np.zeros((1, len(categories))), columns=categories)
            category_data[category] = 1  # Set the current category to 1

            # Concatenate numeric and category features
            input_data = pd.concat([input_data, category_data], axis=1)

            # Ensure the order of columns matches the original training data
            input_data = input_data[feature_columns]

            # Scale the input data using the saved scaler
            input_scaled = scaler.transform(input_data)

            # Predict the amount
            predicted_amount_scaled = model.predict(input_scaled)
            predicted_amount = scaler_target.inverse_transform([[predicted_amount_scaled[0]]])[0][0]
            category_total += predicted_amount

        predictions[category] = category_total
        total_monthly_expense += category_total

    # Return predictions as JSON
    return jsonify({
        'category_wise_predictions': predictions,
        'total_monthly_expense': total_monthly_expense
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the provided port or default to 5000
    app.run(host='0.0.0.0', port=port)
