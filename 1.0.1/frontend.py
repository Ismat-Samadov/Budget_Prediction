import streamlit as st
import requests
import json
from datetime import datetime

# Define the URL of the Flask API
URL = "http://127.0.0.1:5000/predict"

st.title('Machine Learning Prediction Interface')

# Creating form for user input
with st.form(key='prediction_form'):
    # Using a date input and a time input
    date_input = st.date_input("Choose Date")
    time_input = st.time_input("Choose Time")

    # Combining date and time input into one datetime
    datetime_input = datetime.combine(date_input, time_input)

    category = st.selectbox("Select Category", options=["Travel", "Clothing", "Coffe", "Communal", "Events", "Film/enjoyment"])
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Prepare the request data
    data = {
        "datetime": datetime_input.strftime('%Y-%m-%d %H:%M:%S'),  # Formatting datetime as string
        "category": category
    }

    # Make a POST request to the Flask API
    response = requests.post(URL, json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction'][0]}")
    else:
        st.error("Failed to get response from the API")
