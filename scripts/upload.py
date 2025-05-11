import hopsworks
import os
import mlflow
import mlflow.sklearn

# Step 1: Read model info
with open('model_info.txt') as f:
    for line in f:
        key, value = line.strip().split('=')
        os.environ[key] = value

# Step 2: Log in to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citi_bikes_project",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

# Step 3: Get the Model Registry
mr = project.get_model_registry()

# Step 4: Check for existing model versions
model_name = "citi_bike_trip_predictor"
existing_models = mr.get_models(name=model_name)
if existing_models:
    latest_version = max([m.version for m in existing_models])
    new_version = latest_version + 1
else:
    new_version = 1

# Step 5: Load and register the model
model = mlflow.sklearn.load_model(f"runs:/{os.getenv('MLFLOW_RUN_ID')}/model")
model_meta = mr.python.create_model(
    name=model_name,
    version=new_version,
    metrics={"mae": float(os.getenv("MODEL_MAE", 0.0))},
    description="LightGBM model for Citi Bike trip prediction"
)
model_meta.save(f"runs:/{os.getenv('MLFLOW_RUN_ID')}/model")
print(f"Model uploaded successfully as version {new_version}.")
