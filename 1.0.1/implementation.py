import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
df = pd.read_excel('07.2022---05.2024.xlsx')

def preprocess_data(df):
    # Convert 'date' and 'time' columns to datetime format separately
    df['datetime'] = pd.to_datetime(df['date'], format='%d.%m.%Y') + pd.to_timedelta(df['time'].astype(str))
    # Convert time values to seconds since midnight
    df['time_seconds'] = df['datetime'].dt.hour * 3600 + df['datetime'].dt.minute * 60 + df['datetime'].dt.second
    # Feature engineering - extract dayofweek
    df['dayofweek'] = df['datetime'].dt.dayofweek

    # One-hot encoding for categorical variable 'category'
    df = pd.get_dummies(df, columns=['category'])

    return df


def predict_expenses(df):
    if df.empty:
        return {}
    X = df[['time_seconds', 'dayofweek'] + [col for col in df.columns if 'category' in col]]
    print("Shape of X:", X.shape)  # Debug print statement
    predictions = model.predict(X)
    category_expenses = dict(zip(df['category'], predictions))
    return category_expenses


def display_results(category_expenses):
    st.write("Predicted Expenses for Each Category:")
    if category_expenses:
        for category, expense in category_expenses.items():
            st.write(f"{category}: ${expense:.2f}")
    else:
        st.write("No data available for the selected month and year.")


def main():
    st.title("Expense Prediction App")

    month = st.slider("Select Month:", 1, 12)
    year = st.number_input("Enter Year:", value=2022, step=1)
    df = pd.read_excel('07.2022---05.2024.xlsx')
    print("Selected Month:", month)  # Debug print statement
    print("Selected Year:", year)  # Debug print statement
    # Preprocess the data
    df = preprocess_data(df)
    print("Unique 'datetime' values in df:", df['datetime'].unique())  # Debug print statement

    filtered_df = df[(df['datetime'].dt.month == month) & (df['datetime'].dt.year == year)]
    # Select only the relevant features used during model training
    filtered_df = filtered_df[['time_seconds', 'dayofweek'] + [col for col in filtered_df.columns if 'category' in col]]

    print("Filtered DataFrame Shape:", filtered_df.shape)  # Debug print statement
    print("Filtered DataFrame Head:", filtered_df.head())  # Debug print statement
    category_expenses = predict_expenses(filtered_df)
    display_results(category_expenses)


if __name__ == "__main__":
    main()
