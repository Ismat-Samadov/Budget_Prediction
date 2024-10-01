from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model, scalers, and label encoder
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler_target = joblib.load('scaler_target.pkl')  # Load the target scaler for inverse scaling

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from the POST request
    
    year = data['year']
    month = data['month']
    
    # Add default values for `day` and `day_of_week` (since we're predicting for the whole month)
    day = 1  # Default day
    day_of_week = 0  # Default day of the week (Monday)

    # Predict expenses for each category and sum them for the entire month
    predictions = {}
    total_monthly_expense = 0  # To sum the predicted amounts for all categories

    categories = label_encoder.classes_
    for category in categories:
        category_encoded = label_encoder.transform([category])[0]
        input_data = np.array([[year, month, day, day_of_week, category_encoded]])  # Now using 5 features
        input_scaled = scaler.transform(input_data)
        predicted_amount_scaled = model.predict(input_scaled)
        
        # Inverse transform the predicted scaled amount back to original scale
        predicted_amount = scaler_target.inverse_transform([[predicted_amount_scaled[0]]])[0][0]
        predictions[category] = predicted_amount
        total_monthly_expense += predicted_amount

    # Return the total predicted expense for the month along with per-category predictions
    return jsonify({
        'category_wise_predictions': predictions,
        'total_monthly_expense': total_monthly_expense
    })

if __name__ == '__main__':
    app.run(debug=True)
