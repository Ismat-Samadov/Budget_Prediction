import streamlit as st
import pandas as pd
import pickle
import json
from datetime import datetime

# Load the saved columns' names
with open('columns_used.json', 'r') as json_file:
    columns_used = json.load(json_file)

# Streamlit app
st.title('Expense Prediction App')

# User input for year and month
year_month = st.text_input("Enter the year and month (e.g., 2024-05):")
try:
    input_date = datetime.strptime(year_month, '%Y-%m')
except ValueError:
    st.error("Please enter the year and month in the format YYYY-MM.")
    st.stop()

# Process the input year and month
input_year = input_date.year
input_month = input_date.month

# Calculate time difference based on the input year and month
current_date = datetime.now()
input_datetime = datetime(input_year, input_month, 1)
time_diff = (current_date - input_datetime).days

# Create DataFrame from input year, month, and time_diff
input_data = {'year': [input_year], 'month': [input_month], 'time_diff': [time_diff]}
for category in columns_used[3:]:  # Exclude 'amount' and 'time_diff' columns
    input_data[category] = [0]  # Initialize all other columns with zeros
input_df = pd.DataFrame(input_data, columns=columns_used)  # Include all columns

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

# Predict expenses for each category
st.write("Predicted Expenses for", year_month)
expenses = {}
for category in columns_used[5:]:  # Exclude non-category columns
    prediction = trained_model.predict(input_df)[0]
    expenses[category] = prediction
    st.write(f'{category}: ${prediction:.2f}')
