import hopsworks
import os
import mlflow
import mlflow.sklearn
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

# Step 2: Set MLflow tracking URI
mlflow.set_tracking_uri("https://c.app.hopsworks.ai/p/1231006/mlflow")  # Match train.py URI
print("MLflow tracking URI set to:", mlflow.get_tracking_uri())

# Step 3: Read model info from model_info.txt
run_id = None
mae = None
with open('model_info.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split('=')
        if key == "MLFLOW_RUN_ID":
            run_id = value
        elif key == "MODEL_MAE":
            mae = float(value)
if run_id is None or mae is None:
    raise ValueError("MLFLOW_RUN_ID or MODEL_MAE not found in model_info.txt")

# Step 4: Get the Model Registry
mr = project.get_model_registry()

# Step 5: Check for existing model versions
model_name = "citi_bike_trip_predictor"
existing_models = mr.get_models(name=model_name)
if existing_models:
    latest_version = max([m.version for m in existing_models])
    new_version = latest_version + 1
else:
    new_version = 1

# Step 6: Load and register the model
print(f"Loading model from runs:/{run_id}/model...")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
model_meta = mr.sklearn.create_model(
    name=model_name,
    version=new_version,
    metrics={"mae": mae},
    description="LightGBM model for Citi Bike trip prediction"
)
model_meta.save(model)  # Save the loaded model object
print(f"Model uploaded successfully as version {new_version} with MAE {mae}.")
