import pandas as pd
from datetime import datetime
from pytz import timezone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    feature_names = X.columns  # Store feature names before scaling for later use
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open('fitted_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def tune_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    logging.info("Tuning and evaluating the model...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                               param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Optimized Root Mean Squared Error (RMSE):", rmse)

    # Check feature importance and match to feature names
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
    print(importance_df.head())

    with open('final_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    return best_model

if os.path.exists('budjet.xlsx'):
    df = pd.read_excel('budjet.xlsx')
    df = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = split_and_scale(df)
    trained_model = tune_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
else:
    logging.error("Failed to find the file: budjet.xlsx")
