import pandas as pd
from datetime import datetime
from pytz import timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
import logging
import json
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df):
    logging.info("Performing data preprocessing...")
    try:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        else:
            logging.info("Datetime column already timezone-aware.")
        current_datetime = datetime.now(timezone('UTC'))
        df['time_diff'] = (current_datetime - df['datetime']).dt.days
        df['hours'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday + 1
        df['year'] = df['datetime'].dt.year
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month
        df.drop(['datetime'], axis=1, inplace=True)
        df = pd.get_dummies(df, columns=['category'], drop_first=True)
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise e
    return df

def split_and_scale(df, target_column='amount', test_size=0.2, random_state=42):
    logging.info("Splitting and scaling the data...")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns.tolist()  # Save feature names for later use
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open('fitted_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)  # Save feature names to a JSON file
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
    logging.info("Training and evaluating the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error (RMSE):", rmse)
    with open('final_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

if os.path.exists('budjet.xlsx'):
    df = pd.read_excel('budjet.xlsx')
    df = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df)
    trained_model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
else:
    logging.error("Failed to find the file: budjet.xlsx")
