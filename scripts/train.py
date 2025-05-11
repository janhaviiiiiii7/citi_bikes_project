import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import mlflow
import mlflow.sklearn
import hopsworks
import os

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Step 1: Log in to Hopsworks
print("Logging in to Hopsworks...")
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="CitiBikeTrip",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
print("Logged in successfully.")

# Step 2: Manually set MLflow tracking URI for Hopsworks
tracking_uri = f"https://c.app.hopsworks.ai/p/1228950/mlflow"
mlflow.set_tracking_uri(tracking_uri)
print("MLflow tracking URI set to:", mlflow.get_tracking_uri())

# Step 3: Load the processed data
df = pd.read_csv('data/processed_trips_top_3.csv')
df['start_hour'] = pd.to_datetime(df['start_hour'])

# Step 4: Create lag features efficiently
all_station_data = []
for station in df['start_station_name'].unique():
    station_data = df[df['start_station_name'] == station].sort_values('start_hour').copy()
    lag_columns = [station_data['trip_count'].shift(lag) for lag in range(1, 673)]
    lag_df = pd.concat([station_data] + [pd.Series(col, name=f'lag_{lag}') for lag, col in enumerate(lag_columns, 1)], axis=1)
    all_station_data.append(lag_df)

df = pd.concat(all_station_data, ignore_index=True)
df = df.dropna()

# Step 5: Prepare features and target
X = df[[f'lag_{i}' for i in range(1, 673)]]
y = df['trip_count']

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Step 7: Set MLflow experiment
mlflow.set_experiment("CitiBikeModels")

# Model 1: Baseline (Mean)
with mlflow.start_run(run_name="Baseline"):
    mean_prediction = np.mean(y_train)
    baseline_predictions = np.full_like(y_test, mean_prediction)
    mae = mean_absolute_error(y_test, baseline_predictions)
    mlflow.log_metric("MAE", mae)
    print(f"Baseline MAE: {mae}")

# Model 2: LightGBM with 28-day lags
with mlflow.start_run(run_name="LightGBM_Full"):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mlflow.log_metric("MAE", mae)
    mlflow.sklearn.log_model(model, "model", input_example=X_train.head(5))
    os.environ["MLFLOW_RUN_ID"] = mlflow.active_run().info.run_id
    os.environ["MODEL_MAE"] = str(mae)
    print(f"LightGBM Full MAE: {mae}")

# Model 3: LightGBM with feature reduction (top 10 features)
selector = SelectKBest(score_func=f_regression, k=10)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

with mlflow.start_run(run_name="LightGBM_Reduced"):
    model = GradientBoostingRegressor()
    model.fit(X_train_reduced, y_train)
    predictions = model.predict(X_test_reduced)
    mae = mean_absolute_error(y_test, predictions)
    mlflow.log_metric("MAE", mae)
    mlflow.sklearn.log_model(model, "model", input_example=X_train_reduced[:5])
    print(f"LightGBM Reduced MAE: {mae}")
