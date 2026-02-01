# ============================
# Model Training Script
# ============================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb
import mlflow

# ----------------------------
# Hugging Face Setup
# ----------------------------
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------
# Project Paths & MLflow Config
# ----------------------------
# Get the base directory of the repository
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define model directory relative to root
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Define and set MLflow tracking directory to prevent Permission Errors
MLFLOW_TRACKING_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)

# Set MLflow to use the local workspace instead of system defaults
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")
mlflow.set_experiment("tourism-package-training")

# ----------------------------
# Data Loading (Hugging Face Dataset)
# ----------------------------
HF_DATASET_REPO = "Sriranjan/Tourism-Package-Pred/data/processed"

print("Loading data from Hugging Face...")
Xtrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytest.csv").values.ravel()

print("Data loaded successfully. Preparing training...")

# ----------------------------
# Feature groups
# ----------------------------
numeric_features = [
    "Age", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome",
    "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

categorical_features = [
    "TypeofContact", "CityTier", "Occupation", "Gender",
    "MaritalStatus", "Passport", "OwnCar",
    "ProductPitched", "Designation"
]

# ----------------------------
# Preprocessing & Model pipeline
# ----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Using XGBClassifier for 'ProdTaken' prediction
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ----------------------------
# Training with MLflow
# ----------------------------
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predictions
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(ytest, y_pred_test),
        "f1_score": f1_score(ytest, y_pred_test),
    }

    mlflow.log_metrics(metrics)
    print(f"Final Metrics: {metrics}")

    # ----------------------------
    # Save model locally
    # ----------------------------
    model_path = os.path.join(MODEL_DIR, "tourism_xgb_model.joblib")
    joblib.dump(best_model, model_path)
    
    # Log the artifact using the local relative path
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model trained and saved locally at {model_path}")

# ----------------------------
# Upload model to Hugging Face
# ----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "Sriranjan/Tourism-Package-Pred"

try:
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="models/tourism_xgb_model.joblib",
        repo_id=repo_id,
        repo_type="dataset", 
        commit_message="Upload trained XGBoost Classifier"
    )
    print("Model successfully uploaded to Hugging Face Hub.")
except Exception as e:
    print(f"Failed to upload to HF: {e}")
