import pandas as pd
import os

# Directory containing the raw data files
data_dir = "data"
output_file = "data/processed_trips_top_3.csv"

# Step 1: Load all CSV files (excluding processed_trips_top_3.csv)
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'processed_trips_top_3.csv']
if not files:
    raise FileNotFoundError("No raw Citi Bike trip data files found in 'data/' directory. Expected files like '2023*-citibike-tripdata.csv'.")

df_list = []
for file in files:
    print(f"Loading file: {file}")
    df = pd.read_csv(file)
    df_list.append(df)

if not df_list:
    raise ValueError("No data loaded. Ensure raw data files are present and not empty.")

df = pd.concat(df_list, ignore_index=True)

# Step 2: Clean and preprocess the data
df = df.dropna(subset=['start_station_name', 'started_at'])
df['started_at'] = pd.to_datetime(df['started_at'])
df['start_hour'] = df['started_at'].dt.floor('H')
df['trip_count'] = 1

# Step 3: Aggregate by station and hour
station_hour_counts = df.groupby(['start_station_name', 'start_hour'])['trip_count'].sum().reset_index()

# Step 4: Select top 3 stations by total trips
total_trips = df.groupby('start_station_name')['trip_count'].sum().reset_index()
top_3_stations = total_trips.nlargest(3, 'trip_count')['start_station_name']
df_top_3 = station_hour_counts[station_hour_counts['start_station_name'].isin(top_3_stations)]

# Step 5: Save the processed data
df_top_3.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")
