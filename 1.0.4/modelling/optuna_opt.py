import pandas as pd
import warnings
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# Load the Excel file
file_path = 'budget.xlsx'
budget_data = pd.read_excel(file_path, sheet_name='budjet')

# Remove expenses greater than 100 AZN
# budget_data = budget_data[budget_data['amount'] <= 100]

# Drop rows with NaN values
budget_data.dropna(inplace=True)

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

# Ensure no NaN values in target variable
X, y = X[y.notna()], y[y.notna()]

# Impute missing values for features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for RandomForest with Optuna
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return r2

# Create an Optuna study and optimize
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=100)

# Get the best parameters
best_params_rf = study_rf.best_params
best_score_rf = study_rf.best_value

print(f"Best Parameters for Random Forest: {best_params_rf}")
print(f"Best R2 Score for Random Forest: {best_score_rf}")

# Train and evaluate the model with the best parameters
best_model_rf = RandomForestRegressor(
    **best_params_rf,
    random_state=42
)
best_model_rf.fit(X_train, y_train)
y_pred_rf = best_model_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, R2: {r2_rf}")

# Define the objective function for Gradient Boosting with Optuna
def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
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

# Define the objective function for XGBoost with Optuna
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

# Create Optuna studies for each model
study_gb = optuna.create_study(direction='maximize')
study_gb.optimize(objective_gb, n_trials=100)
print(f"Best Parameters for Gradient Boosting: {study_gb.best_params}")
print(f"Best R2 Score for Gradient Boosting: {study_gb.best_value}")

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=100)
print(f"Best Parameters for XGBoost: {study_xgb.best_params}")
print(f"Best R2 Score for XGBoost: {study_xgb.best_value}")

# Evaluate Linear Regression as a baseline
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print(f"Linear Regression - MAE: {mae_lr}, MSE: {mse_lr}, R2: {r2_lr}")

# Save the results to a DataFrame and print

# Store the results
results = {
    'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Linear Regression'],
    'Best Parameters': [best_params_rf, study_gb.best_params, study_xgb.best_params, 'N/A'],
    'R2 Score': [r2_rf, study_gb.best_value, study_xgb.best_value, r2_lr],
    'MAE': [mae_rf, 'N/A', 'N/A', mae_lr],
    'MSE': [mse_rf, 'N/A', 'N/A', mse_lr]
}

# Create a DataFrame
results_df = pd.DataFrame(results)

# Print the DataFrame
print(results_df)
