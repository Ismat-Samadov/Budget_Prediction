import pandas as pd
import warnings
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the Excel file
file_path = 'budjet.xlsx'
budget_data = pd.read_excel(file_path, sheet_name='budjet')

# Ensure the 'date_time' column is present
if 'date_time' not in budget_data.columns:
    raise KeyError("The 'date_time' column is not found in the DataFrame.")


# Drop rows with NaN values
budget_data.dropna(inplace=True)

# Extracting time-related features
budget_data['date_time'] = pd.to_datetime(budget_data['date_time'])
budget_data['day_of_week'] = budget_data['date_time'].dt.dayofweek
budget_data['month'] = budget_data['date_time'].dt.month
budget_data['hour'] = budget_data['date_time'].dt.hour

# One-hot encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_categories = encoder.fit_transform(budget_data[['category']])

# Save valid categories for later use
valid_categories = budget_data['category'].unique().tolist()

# Create a DataFrame for the encoded categories
encoded_categories_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(['category']))

# Concatenate the original DataFrame with the encoded categories
budget_data_encoded = pd.concat([budget_data, encoded_categories_df], axis=1)
budget_data_encoded.drop(['category', 'date_time'], axis=1, inplace=True)

# Splitting the data into training and testing sets
X = budget_data_encoded.drop('amount', axis=1)
y = budget_data_encoded['amount']

# Ensure no NaN values in target variable
X, y = X[y.notna()], y[y.notna()]

# Save the feature names before imputation
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Impute missing values for features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Gradient Boosting with Optuna
def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

# Create an Optuna study and optimize
study_gb = optuna.create_study(direction='maximize')
study_gb.optimize(objective_gb, n_trials=200)

# Get the best parameters
best_params_gb = study_gb.best_params
best_score_gb = study_gb.best_value

print(f"Best Parameters for Gradient Boosting: {best_params_gb}")
print(f"Best R2 Score for Gradient Boosting: {best_score_gb}")

# Train and evaluate the model with the best parameters
best_model_gb = GradientBoostingRegressor(
    **best_params_gb,
    random_state=42
)
best_model_gb.fit(X_train, y_train)
y_pred_gb = best_model_gb.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Gradient Boosting - MAE: {mae_gb}, MSE: {mse_gb}, R2: {r2_gb}")

# Save the model, encoders, imputers, and valid categories
joblib.dump(best_model_gb, 'best_model_gb.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(valid_categories, 'valid_categories.pkl')
