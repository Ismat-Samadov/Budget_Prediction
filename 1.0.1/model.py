import pandas as pd
from pytz import timezone
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
import logging
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess_data(df):
    """
    Preprocess the DataFrame by converting date and time columns,
    extracting datetime features, and performing one-hot encoding.
    """
    logging.info("Performing data preprocessing...")
    # Convert 'date' and 'time' columns to datetime format
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = pd.to_timedelta(df['time'].astype(str))
    # Combine 'date' and 'time' columns into a single datetime column
    df['datetime'] = df['date'] + df['time']
    # Make the datetime column timezone-aware
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC')
    # Calculate 'time_diff' from current datetime
    current_datetime = datetime.now(timezone('UTC'))
    df['time_diff'] = (current_datetime - df['datetime']).dt.days
    # Extract additional datetime features
    df['hours'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday + 1
    df['year'] = df['datetime'].dt.year
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    # Calculate 'time_seconds' from the 'time' column
    df['time_seconds'] = df['time'].apply(lambda x: x.total_seconds())
    # Drop unnecessary columns
    df.drop(['date', 'time', 'datetime'], axis=1, inplace=True)
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['category'], drop_first=True)
    # Save columns used for training as JSON
    columns_used = df.columns.tolist()
    columns_used.remove('amount')  # Remove 'amount' column
    with open('columns_used.json', 'w') as json_file:
        json.dump(columns_used, json_file)
    return df

def split_and_scale(df, target_column='amount', test_size=0.2, random_state=42):
    """
    Split the DataFrame into features and target variables,
    then scale the features using StandardScaler.
    """
    logging.info("Splitting and scaling the data...")
    # Split data into features (X) and target (y)
    X = df.drop(target_column, axis=1)  # Exclude the target column
    y = df[target_column]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open('fitted_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train a RandomForestRegressor model, evaluate its performance,
    and save the trained model to a file.
    """
    logging.info("Training and evaluating the model...")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error (RMSE):", rmse)
    with open('final_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model

# Read the dataset
df = pd.read_excel('07.2022---05.2024.xlsx')
# Perform preprocessing
df = preprocess_data(df)
# Split and scale the data
X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df)
# Train and evaluate the model
trained_model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
