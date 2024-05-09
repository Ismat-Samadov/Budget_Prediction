import streamlit as st
import requests
from datetime import datetime

# Function to fetch predictions
def fetch_predictions(month, year):
    date_str = f"{month:02d}.{year}"
    url = f"https://budget-72z9.onrender.com/predict?date={date_str}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Streamlit user interface
st.title('Expense Prediction Dashboard')

# User inputs
current_year = datetime.now().year
years = list(range(current_year - 10, current_year + 10))  # Adjust range as needed
months = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

selected_month = st.selectbox('Select Month', list(months.keys()), index=months[datetime.now().strftime('%B')] - 1)
selected_year = st.selectbox('Select Year', years, index=years.index(current_year))

if st.button('Predict Expenses'):
    results = fetch_predictions(months[selected_month], selected_year)
    if results:
        st.write('Predicted Expenses by Category:')
        for category, expense in results.get('category_expenses', {}).items():
            st.write(f"{category}: ${expense:.2f}")
        st.write(f"Total Predicted Expense: ${results.get('total_predicted_expense', 0):.2f}")
    else:
        st.error('Failed to fetch predictions. Check if the API is running.')

