from flask import Flask, request, jsonify
import joblib
import numpy as np
import calendar

# Load the saved model, scalers, and label encoder
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler_target = joblib.load('scaler_target.pkl')  # Load the target scaler (MinMaxScaler)

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

    categories = label_encoder.classes_
    
    # Iterate over each category
    for category in categories:
        category_encoded = label_encoder.transform([category])[0]
        category_total = 0
        
        # Iterate over each day of the month
        for day in range(1, num_days + 1):
            day_of_week = calendar.weekday(year, month, day)  # Get the day of the week
            input_data = np.array([[year, month, day, day_of_week, category_encoded]])  # Use 5 features including day and day_of_week
            
            # Debugging step to ensure correct input before scaling
            print(f"Original input before scaling: {input_data}")
            
            input_scaled = scaler.transform(input_data)
            
            # Debugging step to check scaled input
            print(f"Scaled input: {input_scaled}")
            
            predicted_amount_scaled = model.predict(input_scaled)
            
            # Debugging step to check predicted scaled amount
            print(f"Predicted (scaled): {predicted_amount_scaled}")
            
            # Inverse transform the predicted scaled amount back to original scale using MinMaxScaler
            predicted_amount = scaler_target.inverse_transform([[predicted_amount_scaled[0]]])[0][0]
            
            # Debugging step to check predicted amount after inverse scaling
            print(f"Predicted amount (after inverse scaling): {predicted_amount}")
            
            category_total += predicted_amount  # Sum daily predictions for the current category

        predictions[category] = category_total  # Store the total for the category
        total_monthly_expense += category_total  # Sum for the whole month across categories

    # Return the total predicted expense for the month along with per-category predictions
    return jsonify({
        'category_wise_predictions': predictions,
        'total_monthly_expense': total_monthly_expense
    })

if __name__ == '__main__':
    app.run(debug=True)
