import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import hopsworks
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Log in to Hopsworks
print("Logging in to Hopsworks...")
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citi_bikes_project",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
print("Logged in successfully.")

# Step 2: Load data from Feature Group
print("Loading data from Feature Group 'citi_bike_trips_fg'...")
fs = project.get_feature_store()
fg = fs.get_feature_group(name="citi_bike_trips_fg", version=1)
df = fg.read()
df['start_hour'] = pd.to_datetime(df['start_hour'])

# Step 3: Create lag features
all_station_data = []
for station in df['start_station_name'].unique():
    station_data = df[df['start_station_name'] == station].sort_values('start_hour').copy()
    lag_columns = [station_data['trip_count'].shift(lag) for lag in range(1, 673)]
    lag_df = pd.concat([station_data] + [pd.Series(col, name=f'lag_{lag}') for lag, col in enumerate(lag_columns, 1)], axis=1)
    all_station_data.append(lag_df)

df = pd.concat(all_station_data, ignore_index=True)
df = df.dropna()

# Step 4: Prepare features and target
X = df[[f'lag_{i}' for i in range(1, 673)]]
y = df['trip_count']

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Ensure 'data' directory exists
os.makedirs('data', exist_ok=True)
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Model 1: Baseline (Mean)
mean_prediction = np.mean(y_train)
baseline_predictions = np.full_like(y_test, mean_prediction)
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print(f"Baseline MAE: {mae_baseline}")

# Model 2: LightGBM with 28-day lags
model_full = LGBMRegressor()
model_full.fit(X_train, y_train)
predictions_full = model_full.predict(X_test)
mae_full = mean_absolute_error(y_test, predictions_full)
print(f"LightGBM Full MAE: {mae_full}")

# Save model locally or to Hopsworks File System
os.makedirs('models', exist_ok=True)
import joblib
joblib.dump(model_full, 'models/lightgbm_full_model.pkl')

# Model 3: LightGBM with feature reduction (top 10 features)
selector = SelectKBest(score_func=f_regression, k=10)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

model_reduced = LGBMRegressor()
model_reduced.fit(X_train_reduced, y_train)
predictions_reduced = model_reduced.predict(X_test_reduced)
mae_reduced = mean_absolute_error(y_test, predictions_reduced)
print(f"LightGBM Reduced MAE: {mae_reduced}")

# Save model
joblib.dump(model_reduced, 'models/lightgbm_reduced_model.pkl')

# Save metrics to a file for monitoring
with open('metrics.txt', 'w') as f:
    f.write(f"Baseline_MAE={mae_baseline}\nLightGBM_Full_MAE={mae_full}\nLightGBM_Reduced_MAE={mae_reduced}\n")
