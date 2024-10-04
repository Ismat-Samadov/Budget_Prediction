from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import calendar

# Load the saved model, scalers, and columns
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
scaler_target = joblib.load('scaler_target.pkl')  # Load the target scaler (MinMaxScaler)
feature_columns = joblib.load('feature_columns.pkl')  # Load the original feature columns for prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from the POST request

    year = data['year']
    month = data['month']

    # Get the number of days in the month
    num_days = calendar.monthrange(year, month)[1]

    # Predict expenses for each category and sum them for the entire month
    predictions = {}
    total_monthly_expense = 0  # To sum the predicted amounts for all categories

    # Dummy-encoded category columns (as per your trained model)
    categories = [col for col in feature_columns if 'category_' in col]
    
    # Iterate over each category
    for category in categories:
        category_total = 0
        
        # Iterate over each day of the month
        for day in range(1, num_days + 1):
            day_of_week = calendar.weekday(year, month, day)  # Get the day of the week

            # Construct input data as DataFrame with the correct feature columns
            input_data = pd.DataFrame([[year, month, day, day_of_week]], columns=['year', 'month', 'day', 'day_of_week'])
            
            # Create a DataFrame for category dummy columns
            category_data = pd.DataFrame(np.zeros((1, len(categories))), columns=categories)
            category_data[category] = 1  # Set the current category column to 1
            
            # Concatenate the numeric features and category dummies
            input_data = pd.concat([input_data, category_data], axis=1)

            # Ensure that the input data matches the order of columns in the original training set
            input_data = input_data[feature_columns]

            # Scale the input data using the saved scaler
            input_scaled = scaler.transform(input_data)

            # Predict the scaled amount
            predicted_amount_scaled = model.predict(input_scaled)

            # Inverse transform the predicted scaled amount back to the original scale using MinMaxScaler
            predicted_amount = scaler_target.inverse_transform([[predicted_amount_scaled[0]]])[0][0]

            # Sum daily predictions for the current category
            category_total += predicted_amount

        predictions[category] = category_total  # Store the total for the category
        total_monthly_expense += category_total  # Sum for the whole month across categories

    # Return the total predicted expense for the month along with per-category predictions
    return jsonify({
        'category_wise_predictions': predictions,
        'total_monthly_expense': total_monthly_expense
    })

if __name__ == '__main__':
    app.run(debug=True)
