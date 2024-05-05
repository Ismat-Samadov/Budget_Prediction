import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_excel('07.2022---05.2024.xlsx')

# Data preprocessing
# Convert categorical variables into numerical format using one-hot encoding
df = pd.get_dummies(df, columns=['category'])

# Feature engineering
# Extract relevant features such as time of day, day of the week, etc.
df['hour'] = df['time'].apply(lambda x: x.hour)  # Extract hour from datetime
df['dayofweek'] = df['date'].dt.dayofweek

# Splitting the data into training and testing sets
X = df[['hour', 'dayofweek'] + [col for col in df.columns if 'category' in col]]
y = df['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)

# Prediction (using future dates as input)
# Load future data
future_df = pd.read_excel('07.2022---05.2024.xlsx')

# Preprocess future data
future_df = pd.get_dummies(future_df, columns=['category'])
future_df['hour'] = future_df['time'].apply(lambda x: x.hour)  # Extract hour from datetime
future_df['dayofweek'] = future_df['date'].dt.dayofweek

# Select features for prediction
future_X = future_df[['hour', 'dayofweek'] + [col for col in future_df.columns if 'category' in col]]

# Make predictions
future_predictions = model.predict(future_X)

# Print predicted expenses
print('Predicted expenses for future dates:')
print(future_predictions)
