import streamlit as st
import requests
import pandas as pd

# Streamlit app title
st.title("Expense Prediction Dashboard")

# User inputs for selecting month and year
year = st.selectbox("Select Year", options=[2023, 2024, 2025], index=1)
month = st.selectbox("Select Month", options=list(range(1, 13)), format_func=lambda x: pd.to_datetime(f"2024-{x}-01").strftime('%B'))

# Button to trigger prediction
if st.button("Predict Expenses"):
    # Make request to the Flask API
    response = requests.get(f"https://budget-analyse.onrender.com/predict?year={year}&month={month}")
    
    if response.status_code == 200:
        predictions = response.json()
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions.items(), columns=["Category", "Predicted Expense"])
        
        # Move total predicted expense to the end
        total_expense = pred_df[pred_df["Category"] == "Total Predicted Expense"]
        pred_df = pred_df[pred_df["Category"] != "Total Predicted Expense"]
        pred_df = pd.concat([pred_df, total_expense], ignore_index=True)
        
        # Display predictions in a table
        st.subheader(f"Predicted Expenses for {pd.to_datetime(f'{year}-{month}-01').strftime('%B %Y')}")
        st.dataframe(pred_df)
        
        # Display total predicted expense
        total_expense_amount = total_expense['Predicted Expense'].values[0]
        st.subheader(f"Total Predicted Expense: ₼{total_expense_amount:.2f}")
    else:
        st.error("Failed to fetch predictions. Please try again.")
