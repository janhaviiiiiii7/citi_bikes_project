import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import mlflow
import mlflow.sklearn
import hopsworks
import os

# Step 1: Log in to Hopsworks
print("Logging in to Hopsworks...")
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citi_bikes_project",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
print("Logged in successfully.")

# Step 2: Set MLflow tracking URI
tracking_uri = f"https://c.app.hopsworks.ai/p/1228950/mlflow"
mlflow.set_tracking_uri(tracking_uri)
print("MLflow tracking URI set to:", mlflow.get_tracking_uri())

# Step 3: Create MLflow experiment if it doesn't exist
experiment_name = "CitiBikeModels"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        print(f"Created MLflow experiment: {experiment_name}")
    else:
        print(f"Using existing MLflow experiment: {experiment_name}")
except Exception as e:
    print(f"Failed to create/access experiment: {e}")
    raise
mlflow.set_experiment(experiment_name)

# Step 4: Load data from Feature Group
print("Loading data from Feature Group 'citi_bike_trips_fg'...")
fs = project.get_feature_store()
fg = fs.get_feature_group(name="citi_bike_trips_fg", version=1)
df = fg.read()
df['start_hour'] = pd.to_datetime(df['start_hour'])

# Step 5: Create lag features
all_station_data = []
for station in df['start_station_name'].unique():
    station_data = df[df['start_station_name'] == station].sort_values('start_hour').copy()
    lag_columns = [station_data['trip_count'].shift(lag) for lag in range(1, 673)]
    lag_df = pd.concat([station_data] + [pd.Series(col, name=f'lag_{lag}') for lag, col in enumerate(lag_columns, 1)], axis=1)
    all_station_data.append(lag_df)

df = pd.concat(all_station_data, ignore_index=True)
df = df.dropna()

# Step 6: Prepare features and target
X = df[[f'lag_{i}' for i in range(1, 673)]]
y = df['trip_count']

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Model 1: Baseline (Mean)
with mlflow.start_run(run_name="Baseline"):
    mean_prediction = np.mean(y_train)
    baseline_predictions = np.full_like(y_test, mean_prediction)
    mae = mean_absolute_error(y_test, baseline_predictions)
    mlflow.log_metric("MAE", mae)
    print(f"Baseline MAE: {mae}")

# Model 2: LightGBM with 28-day lags
with mlflow.start_run(run_name="LightGBM_Full"):
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mlflow.log_metric("MAE", mae)
    mlflow.sklearn.log_model(model, "model", input_example=X_train.head(5))
    run_id = mlflow.active_run().info.run_id
    with open('model_info.txt', 'w') as f:
        f.write(f"MLFLOW_RUN_ID={run_id}\nMODEL_MAE={mae}")
    print(f"LightGBM Full MAE: {mae}")

# Model 3: LightGBM with feature reduction (top 10 features)
selector = SelectKBest(score_func=f_regression, k=10)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

with mlflow.start_run(run_name="LightGBM_Reduced"):
    model = LGBMRegressor()
    model.fit(X_train_reduced, y_train)
    predictions = model.predict(X_test_reduced)
    mae = mean_absolute_error(y_test, predictions)
    mlflow.log_metric("MAE", mae)
    mlflow.sklearn.log_model(model, "model", input_example=X_train_reduced[:5])
    print(f"LightGBM Reduced MAE: {mae}")
