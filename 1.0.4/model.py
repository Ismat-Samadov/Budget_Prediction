import pandas as pd
from datetime import datetime
from pytz import timezone
from sklearn.model_selection import train_test_split, GridSearchCV
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
    with open('/Users/ismatsamadov/Budget_Analyse/1.0.4/fitted_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open('/Users/ismatsamadov/Budget_Analyse/1.0.4/feature_names.json', 'w') as f:
        json.dump(feature_names, f)  # Save feature names to a JSON file
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    logging.info("Training and evaluating the model...")
    
    # Define the model and hyperparameter search space
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Retrieve the best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    
    # Predict and evaluate the model
    y_pred = best_model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Save the best model
    with open('/Users/ismatsamadov/Budget_Analyse/1.0.4/final_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    
    # Save mean and std values for normalization
    mean_values = X_train.mean().to_dict()
    std_values = X_train.std().to_dict()
    mean_std_values_path = '/Users/ismatsamadov/Budget_Analyse/1.0.4/mean_std_values.json'
    with open(mean_std_values_path, 'w') as f:
        json.dump({'mean': mean_values, 'std': std_values}, f)
    logging.info(f"Mean and Std values saved to {mean_std_values_path}")

    return best_model

if os.path.exists('/Users/ismatsamadov/Budget_Analyse/1.0.4/budget.xlsx'):
    df = pd.read_excel('/Users/ismatsamadov/Budget_Analyse/1.0.4/budget.xlsx')
    df = preprocess_data(df)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df)
    trained_model = train_and_evaluate_model(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
else:
    logging.error("Failed to find the file: budget.xlsx")
