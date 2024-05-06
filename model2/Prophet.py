# from fbprophet import Prophet
#
# # Train Prophet model for each category separately
# prophet_models = {}
# for category in df.columns[1:]:
#     df_prophet = df[[category]].reset_index().rename(columns={'datetime':'ds', category:'y'})
#     model = Prophet()
#     model.fit(df_prophet)
#     prophet_models[category] = model
#
# # Forecast using Prophet model
# prophet_forecasts = {}
# for category, model in prophet_models.items():
#     future = model.make_future_dataframe(periods=30, freq='D')
#     forecast = model.predict(future)
#     prophet_forecasts[category] = forecast[['ds', 'yhat']].set_index('ds').rename(columns={'yhat':category})

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_excel('07.2022---05.2024.xlsx')

# Data preprocessing
df = pd.get_dummies(df, columns=['category'])
df['hour'] = df['time'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek

# Splitting the data into training and testing sets
X = df[['hour', 'dayofweek'] + [col for col in df.columns if 'category' in col]]
y = df['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regressor
model = RandomForestRegressor()
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Tuned Random Forest Regressor Root Mean Squared Error:", rmse)

# Feature selection with SelectFromModel
selector = SelectFromModel(best_model)
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Final model with selected features
final_model = RandomForestRegressor()
final_model.fit(X_train_selected, y_train)
y_pred_final = final_model.predict(X_test_selected)
rmse_final = mean_squared_error(y_test, y_pred_final, squared=False)
print("Selected Features Random Forest Regressor Root Mean Squared Error:", rmse_final)
