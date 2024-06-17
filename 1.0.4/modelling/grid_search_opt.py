import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the Excel file
file_path = 'budget.xlsx'
budget_data = pd.read_excel(file_path, sheet_name='budjet')

# Extracting time-related features
budget_data['datetime'] = pd.to_datetime(budget_data['datetime'])
budget_data['day_of_week'] = budget_data['datetime'].dt.dayofweek
budget_data['month'] = budget_data['datetime'].dt.month
budget_data['hour'] = budget_data['datetime'].dt.hour

# One-hot encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_categories = encoder.fit_transform(budget_data[['category']])

# Create a DataFrame for the encoded categories
encoded_categories_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(['category']))

# Concatenate the original DataFrame with the encoded categories
budget_data_encoded = pd.concat([budget_data, encoded_categories_df], axis=1)
budget_data_encoded.drop(['category', 'datetime'], axis=1, inplace=True)

# Splitting the data into training and testing sets
X = budget_data_encoded.drop('amount', axis=1)
y = budget_data_encoded['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params_rf = grid_search.best_params_
best_score_rf = grid_search.best_score_

print(f"Best Parameters: {best_params_rf}")
print(f"Best Score: {best_score_rf}")

# Train and evaluate the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")
