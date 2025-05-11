import hopsworks
import os
import joblib
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

# Step 2: Get the Model Registry
mr = project.get_model_registry()

# Step 3: Check for existing model versions
model_name = "citi_bike_trip_predictor"
existing_models = mr.get_models(name=model_name)
if existing_models:
    latest_version = max([m.version for m in existing_models])
    new_version = latest_version + 1
else:
    new_version = 1

# Step 4: Load and register the model (using LightGBM Full as an example)
model_path = 'models/lightgbm_full_model.pkl'
if os.path.exists(model_path):
    # Verify the model can be loaded (optional)
    model = joblib.load(model_path)
    # Read MAE from metrics.txt
    with open('metrics.txt', 'r') as f:
        lines = f.readlines()
        mae = float([line.split('=')[1] for line in lines if 'LightGBM_Full_MAE' in line][0])
    # Create model metadata
    model_meta = mr.sklearn.create_model(
        name=model_name,
        version=new_version,
        metrics={"mae": mae},
        description="LightGBM model for Citi Bike trip prediction"
    )
    # Save the model file path (not the model object)
    model_meta.save(model_path)
    print(f"Model uploaded successfully as version {new_version} with MAE {mae}.")
else:
    raise FileNotFoundError(f"Model file {model_path} not found.")
