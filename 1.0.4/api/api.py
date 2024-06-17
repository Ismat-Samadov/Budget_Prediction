from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your trained model, encoders, imputers, feature names, and valid categories
model = joblib.load('best_model_gb.pkl')
encoder = joblib.load('encoder.pkl')
imputer = joblib.load('imputer.pkl')
feature_names = joblib.load('feature_names.pkl')
valid_categories = joblib.load('valid_categories.pkl')

def generate_features(month, category):
    # Create a DataFrame with the same structure as your training data
    future_data = pd.DataFrame({'month': [month], 'day_of_week': [0], 'hour': [0]})
    
    # One-hot encode the category
    category_encoded = encoder.transform([[category]])
    category_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['category']))
    
    # Concatenate the category data to future data
    future_data = pd.concat([future_data, category_df], axis=1)
    
    # Ensure the order of columns is consistent with training data
    future_data = future_data.reindex(columns=feature_names, fill_value=0)
    
    # Apply transformations
    future_data = imputer.transform(future_data)
    
    return future_data

@app.route('/predict', methods=['GET'])
def predict():
    try:
        year = request.args.get('year')
        month = request.args.get('month')
        
        if year is None or month is None:
            return jsonify({'error': 'Missing year or month query parameters'}), 400
        
        year = int(year)
        month = int(month)
        
        predictions = {}
        total_predicted_expense = 0
        
        for category in valid_categories:
            features = generate_features(month, category)
            prediction = model.predict(features)[0]
            predictions[category] = prediction
            total_predicted_expense += prediction
        
        predictions['Total Predicted Expense'] = total_predicted_expense
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
