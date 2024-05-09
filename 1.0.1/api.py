import pandas as pd
from datetime import datetime
from pytz import timezone
from flask import Flask, request, jsonify
import pickle
import logging
import json
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_scaler_and_features():
    with open('final_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('fitted_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


model, scaler, feature_names = load_model_scaler_and_features()


def preprocess_data(data_frame, feature_names):
    data_frame['datetime'] = pd.to_datetime(data_frame['datetime'], errors='coerce')
    data_frame = data_frame.dropna(subset=['datetime'])
    if data_frame['datetime'].dt.tz is None:
        data_frame['datetime'] = data_frame['datetime'].dt.tz_localize('UTC')

    current_datetime = datetime.now(timezone('UTC'))
    data_frame['time_diff'] = (current_datetime - data_frame['datetime']).dt.days
    data_frame['hours'] = data_frame['datetime'].dt.hour
    data_frame['weekday'] = data_frame['datetime'].dt.weekday + 1
    data_frame['year'] = data_frame['datetime'].dt.year
    data_frame['day_of_year'] = data_frame['datetime'].dt.dayofyear
    data_frame['month'] = data_frame['datetime'].dt.month
    data_frame.drop(['datetime'], axis=1, inplace=True)
    data_frame = pd.get_dummies(data_frame, columns=['category'], drop_first=True)

    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in data_frame.columns:
            data_frame[feature] = 0

    # Reorder columns as per the training data
    data_frame = data_frame[feature_names]

    scaled_features = scaler.transform(data_frame)
    return scaled_features


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    new_data = pd.DataFrame([data])
    prepared_data = preprocess_data(new_data, feature_names)
    prediction = model.predict(prepared_data)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
