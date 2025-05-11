import hopsworks
import pandas as pd
import requests
import json
import os

# Step 1: Log in to Hopsworks
print("Logging in to Hopsworks...")
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="CitiBikeTrip",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
print("Logged in successfully.")

# Step 2: Get the Feature Store
print("Getting Feature Store...")
fs = project.get_feature_store()
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

# Step 5: Get the deployed model's serving endpoint
serving_url = "https://citibiketrippredictor.citibiketrip.hopsworks.ai/v1/models/citibiketrippredictor:predict"
print("Using serving URL:", serving_url)

# Step 6: Generate predictions by calling the serving endpoint
print("Generating predictions...")
predictions = []
for i in range(len(X)):
    input_data = X.iloc[i].values.tolist()
    payload = {"instances": [input_data]}
    print(f"Sending request for row {i}: {input_data[:5]}...")  # Print first 5 features for debugging
    response = requests.post(serving_url, json=payload, headers={"Authorization": f"ApiKey {os.getenv('HOPSWORKS_API_KEY')}"})
    if response.status_code == 200:
        prediction = response.json()["predictions"][0]
        predictions.append(prediction)
        print(f"Prediction for row {i}: {prediction}")
    else:
        print(f"Failed to get prediction for row {i}: {response.status_code} - {response.text}")
        raise Exception(f"Prediction failed: {response.status_code} - {response.text}")

# Step 7: Add predictions to the DataFrame
df['predicted_trip_count'] = predictions
print("Predictions added to DataFrame, shape:", df.shape)

# Step 8: Save predictions to a new Feature Group
print("Saving predictions to Feature Group 'citi_bike_predictions_fg'...")
prediction_fg = fs.get_or_create_feature_group(
    name="citi_bike_predictions_fg",
    version=1,
    description="Predicted trip counts for top 3 stations",
    primary_key=['start_station_name', 'start_hour']
)
prediction_fg.insert(df[['start_station_name', 'start_hour', 'predicted_trip_count']], write_options={'wait': True})
print("Predictions saved successfully.")
