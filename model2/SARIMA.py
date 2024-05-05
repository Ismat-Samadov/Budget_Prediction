import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def read_data(file_path):
    # Read the data from the specified Excel file
    df = pd.read_excel(file_path)
    return df


def preprocess_data(df):
    # Convert 'date' and 'time' columns to datetime format separately
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.time

    # Combine 'date' and 'time' columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')

    # Drop 'date' and 'time' columns
    df = df.drop(['date', 'time'], axis=1)

    # Sort the dataframe by the datetime index
    df = df.sort_values(by='datetime')

    # Set 'datetime' column as index
    df.set_index('datetime', inplace=True)

    # Set the frequency of the datetime index
    df.index.freq = pd.infer_freq(df.index)

    return df


def train_sarima_model(df, order, seasonal_order, freq='MS'):
    # Train SARIMA model for each category separately
    sarima_models = {}
    for category in df.columns[1:]:
        model = SARIMAX(df[category], order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                        enforce_invertibility=False)

        # Explicitly set the frequency of the datetime index
        model = model.fit(start_params=np.random.random(size=model.k_params), method='powell', maxiter=1000, disp=False,
                          freq=freq)

        sarima_models[category] = model
    return sarima_models


def forecast_next_month(sarima_models, df):
    # Forecast using SARIMA model for the next month
    sarima_forecasts = {}
    for category, model in sarima_models.items():
        forecast = model.forecast(steps=1)
        sarima_forecasts[category] = forecast.iloc[0]  # Access the first forecasted value

    # Create DataFrame for forecasts
    sarima_forecasts_df = pd.DataFrame.from_dict(sarima_forecasts, orient='index', columns=['Forecast'])
    sarima_forecasts_df.index.name = 'Category'  # Set index name to 'Category'

    # Drop rows with NaN forecasts
    sarima_forecasts_df = sarima_forecasts_df.dropna()

    # Convert index to datetime
    sarima_forecasts_df.index = pd.to_datetime(sarima_forecasts_df.index, errors='coerce')

    return sarima_forecasts_df


def get_monthly_estimates(sarima_forecasts_df):
    # Fill NaT values with NaN
    sarima_forecasts_df = sarima_forecasts_df.fillna(pd.NA)

    if not sarima_forecasts_df.empty:
        # Check if index contains datetime values
        if isinstance(sarima_forecasts_df.index, pd.DatetimeIndex):
            # Drop rows with NaN values
            sarima_forecasts_df = sarima_forecasts_df.dropna()

            if not sarima_forecasts_df.empty:  # Check if DataFrame is not empty after dropping NaNs
                # Remove any NaT values from the index
                sarima_forecasts_df = sarima_forecasts_df[~sarima_forecasts_df.index.isna()]

                # Check if there are any valid datetime values remaining
                if not sarima_forecasts_df.empty:
                    # Print DataFrame before resampling
                    print("DataFrame before resampling:")
                    print(sarima_forecasts_df)

                    # Calculate monthly estimates for each category as the sum of forecasted values
                    monthly_sum = sarima_forecasts_df.resample('M').sum()
                    return monthly_sum
                else:
                    print("No valid datetime values after removing NaT values. Cannot calculate monthly estimates.")
                    return None
            else:
                print("No valid data after dropping NaN values. Cannot calculate monthly estimates.")
                return None
        else:
            print("Index does not contain datetime values. Cannot resample.")
            return None
    else:
        print("No valid data to calculate monthly estimates.")
        return None


def get_forecasts_for_month(month, sarima_forecasts_df):
    # Filter forecasts for the given month
    forecasts_for_month = sarima_forecasts_df.loc[month]

    # Convert the forecasts to a dictionary
    forecasts_dict = forecasts_for_month.to_dict()

    # Remove the 'category_' prefix from the keys and handle IndexError
    forecasts_dict = {category.split('_')[1]: value for category, value in forecasts_dict.items() if
                      len(category.split('_')) > 1}

    return forecasts_dict


if __name__ == "__main__":
    # Define SARIMA parameters (p, d, q) and (P, D, Q, S)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # Read the data
    df = read_data('07.2022---05.2024.xlsx')

    # Preprocess the data
    df = preprocess_data(df)

    # Train SARIMA model
    sarima_models = train_sarima_model(df, order, seasonal_order)

    # Forecast for the next month
    sarima_forecasts_df = forecast_next_month(sarima_models, df)

    # Display forecasts for the next month
    print("SARIMA Forecasts for the Next Month:")
    print(sarima_forecasts_df.tail(3))

    # Calculate monthly estimates
    monthly_sum = get_monthly_estimates(sarima_forecasts_df)
    print("\nMonthly Estimates for Each Category (Sum of Forecasted Values):")
    print(monthly_sum)

    # Example usage: Get forecasts for a specific month
    month = '2024-05'  # Specify the month for which you want to get forecasts
    forecasts_for_month = get_forecasts_for_month(month, sarima_forecasts_df)

    # Print the forecasts for the specified month
    for category, value in forecasts_for_month.items():
        print(f"{category}: {value}")
