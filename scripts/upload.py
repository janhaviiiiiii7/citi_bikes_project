import hopsworks
import os
import mlflow
import mlflow.sklearn

# Step 1: Log in to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="CitiBikeTrip",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

# Step 2: Get the Model Registry
mr = project.get_model_registry()

# Step 3: Check for existing model versions
model_name = "citi_bike_trip_predictor"
existing_models = mr.get_models(name=model_name)
if existing_models:
    latest_version = max([m.version for m in existing_models])
    print(f"Latest model version found: {latest_version}")
    new_version = latest_version + 1
else:
    print("No existing models found. Starting with version 1.")
    new_version = 1

# Step 4: Log the model to MLflow and upload to Hopsworks
with mlflow.start_run():
    # Assuming the model is already logged in MLflow during train.py
    model = mlflow.sklearn.load_model("runs:/{}/model".format(os.getenv("MLFLOW_RUN_ID")))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )

# Step 5: Register the model in Hopsworks with the new version
model_meta = mr.python.create_model(
    name=model_name,
    version=new_version,
    metrics={"mae": float(os.getenv("MODEL_MAE", 0.0))},
    description="Gradient Boosting model for Citi Bike trip prediction"
)
model_meta.save("runs:/{}/model".format(os.getenv("MLFLOW_RUN_ID")))

print(f"Model uploaded successfully as version {new_version}.")
