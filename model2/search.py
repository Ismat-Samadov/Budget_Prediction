import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import root_mean_squared_error  # Import root_mean_squared_error

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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a list of regression models
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor()
]

# Evaluate each model
for model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, y_pred)  # Use root_mean_squared_error
    print(model.__class__.__name__, 'Root Mean Squared Error:', rmse)

    # Hyperparameter tuning with GridSearchCV
    if isinstance(model, DecisionTreeRegressor) or isinstance(model, RandomForestRegressor):
        param_grid = {'max_depth': [None, 10, 20],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_test_scaled)
        rmse = root_mean_squared_error(y_test, y_pred)  # Use root_mean_squared_error
        print("Tuned", model.__class__.__name__, 'Root Mean Squared Error:', rmse)

    # Feature selection with SelectFromModel
    if isinstance(model, RandomForestRegressor):
        selector = SelectFromModel(model)
        selector.fit(X_train_scaled, y_train)
        X_train_selected = selector.transform(X_train_scaled)
        X_test_selected = selector.transform(X_test_scaled)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        rmse = root_mean_squared_error(y_test, y_pred)  # Use root_mean_squared_error
        print("Selected Features", model.__class__.__name__, 'Root Mean Squared Error:', rmse)
