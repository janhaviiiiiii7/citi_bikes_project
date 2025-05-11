import hopsworks
import pandas as pd
import os
import joblib

# Step 1: Log in to Hopsworks
print("Logging in to Hopsworks...")
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citi_bikes_project",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
print("Logged in successfully.")

# Step 2: Get the Feature Store and Model Registry
print("Getting Feature Store and Model Registry...")
fs = project.get_feature_store()
mr = project.get_model_registry()
print("Feature Store retrieved:", fs)

# Step 3: Load the latest data from the Feature Group
print("Loading data from Feature Group 'citi_bike_trips_fg'...")
fg = fs.get_feature_group(name="citi_bike_trips_fg", version=1)
df = fg.read()
print("Data loaded, shape:", df.shape)
df['start_hour'] = pd.to_datetime(df['start_hour'])

# Step 4: Create lag features (same as training)
print("Creating lag features...")
all_station_data = []
for station in df['start_station_name'].unique():
    station_data = df[df['start_station_name'] == station].sort_values('start_hour').copy()
    lag_columns = [station_data['trip_count'].shift(lag) for lag in range(1, 673)]
    lag_df = pd.concat([station_data] + [pd.Series(col, name=f'lag_{lag}') for lag, col in enumerate(lag_columns, 1)], axis=1)
    all_station_data.append(lag_df)

df = pd.concat(all_station_data, ignore_index=True)
df = df.dropna()
X = df[[f'lag_{i}' for i in range(1, 673)]]
print("Lag features created, X shape:", X.shape)

# Step 5: Load the model from Hopsworks Model Registry
print("Loading model from Model Registry...")
try:
    model_meta = mr.get_model("citi_bike_trip_predictor", version=1)
    model_path = model_meta.download()
    model = joblib.load(os.path.join(model_path, "model.pkl"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

# Step 6: Generate predictions
print("Generating predictions...")
predictions = model.predict(X)
df['predicted_trip_count'] = predictions
print("Predictions added to DataFrame, shape:", df.shape)

# Step 7: Save predictions to a new Feature Group
print("Saving predictions to Feature Group 'citi_bike_predictions_fg'...")
prediction_fg = fs.get_or_create_feature_group(
    name="citi_bike_predictions_fg",
    version=1,
    description="Predicted trip counts for top 3 stations",
    primary_key=['start_station_name', 'start_hour'],
    online_enabled=False
)
prediction_fg.insert(df[['start_station_name', 'start_hour', 'predicted_trip_count']], write_options={'wait_for_job': True})
print("Predictions saved successfully.")
