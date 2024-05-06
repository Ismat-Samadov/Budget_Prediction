import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
df = pd.read_excel('07.2022---05.2024.xlsx')

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time

# Convert time values to seconds since midnight
df['time_seconds'] = df['time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# Data preprocessing
# Convert categorical variables into numerical format using one-hot encoding
df = pd.get_dummies(df, columns=['category'])

# Feature engineering
df['dayofweek'] = df['date'].dt.dayofweek

# Splitting the data into training and testing sets
X = df[['time_seconds', 'dayofweek'] + [col for col in df.columns if 'category' in col]]
y = df['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regressor with Tuning
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Tuned RandomForestRegressor RMSE:", rmse)

# Feature selection with SelectFromModel
selector = SelectFromModel(model, threshold='median')
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Final 1.0.0 with selected features
final_model = RandomForestRegressor(n_estimators=200, random_state=42)
final_model.fit(X_train_selected, y_train)
y_pred_final = final_model.predict(X_test_selected)
rmse_final = mean_squared_error(y_test, y_pred_final, squared=False)
print("Selected Features RMSE:", rmse_final)

# Save the final 1.0.0 to a .pkl file
with open('final_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)

