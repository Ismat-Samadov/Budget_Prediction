from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os  # Needed for port environment variable

# Load the saved model, scalers, and feature columns
model = joblib.load('xgb_model.pkl')  # Updated to XGBoost model
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

    predictions = {}

    # Dummy-encoded category columns
    categories = [col for col in feature_columns if 'category_' in col]

    # Create input DataFrame for the monthly prediction
    input_df = pd.DataFrame(columns=feature_columns)

    # Set the year and month
    input_df['year'] = [year]
    input_df['month'] = [month]

    # Predict for each category
    for category in categories:
        # Set the current category column to 1 and others to 0
        input_df[category] = 1
        for other_category in categories:
            if other_category != category:
                input_df[other_category] = 0

        # Ensure the order of columns matches the original training data
        input_df = input_df[feature_columns]

        # Scale the input data using the saved scaler
        input_scaled = scaler.transform(input_df)

        # Predict the scaled amount using XGBoost model
        predicted_amount_scaled = model.predict(input_scaled)

        # Inverse transform the predicted scaled amount back to original scale using the target scaler
        predicted_amount = scaler_target.inverse_transform([[predicted_amount_scaled[0]]])[0][0]

        # Store the predicted amount for the category
        predictions[category] = predicted_amount

    # Return predictions as JSON
    return jsonify({
        'category_wise_predictions': predictions,
        'total_monthly_expense': sum(predictions.values())
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 500))  # Use the provided port or default to 5001
    app.run(host='0.0.0.0', port=port)
