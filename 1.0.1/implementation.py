import streamlit as st
import pandas as pd
import pickle
import json
import datetime
from pytz import timezone
from sklearn.preprocessing import StandardScaler
import logging
# Read the categories from JSON
with open('categories.json', 'r') as json_file:
    categories = json.load(json_file)

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

# Preprocess function
def preprocess_data(df, categories):
    """
    Preprocess the DataFrame by converting date and time columns,
    extracting datetime features, and performing one-hot encoding.
    """
    logging.info("Performing data preprocessing...")
    # Extract datetime features from the input date
    df['hours'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday + 1
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    # Calculate 'time_seconds' from the 'date' column
    df['time_seconds'] = df['date'].apply(lambda x: x.timestamp())
    # Drop unnecessary columns
    df.drop(['date'], axis=1, inplace=True)
    # Add missing categories as columns filled with zeros
    missing_categories = set(categories) - set(df.columns)
    for category in missing_categories:
        df[category] = 0
    return df

# Streamlit app
st.title('Expense Prediction App')

# User input
input_date = st.date_input("Select a date:", datetime.datetime.now().date())

# Create DataFrame from input data
input_data = {'date': pd.to_datetime(input_date)}
input_df = pd.DataFrame([input_data])

# Preprocess input data
input_df = preprocess_data(input_df, categories)

# Standardize the features
scaler = StandardScaler()
input_df_scaled = scaler.fit_transform(input_df)

# Predict expenses for each category
st.write("Predicted Expenses:")
for category in categories:
    prediction = trained_model.predict(input_df_scaled)[0]
    st.write(f'{category}: ${prediction:.2f}')
