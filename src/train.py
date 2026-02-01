# ============================
# Model Training Script
# ============================

# ----------------------------
# Data & ML libraries
# ----------------------------
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import mlflow

# ----------------------------
# Hugging Face (MODEL hosting only)
# ----------------------------
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------
# MLflow setup
# ----------------------------
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-training")

# ----------------------------
# Paths (Hugging Face Dataset)
# ----------------------------
HF_DATASET_REPO = "Sriranjan/Tourism-Package-Pred/data/processed"


Xtrain = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv"
)
Xtest = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv"
)

ytrain = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv"
).values.ravel()

ytest = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/ytest.csv"
).values.ravel()

print("Data loaded successfully from Hugging Face")



# Get the base directory of the repository
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# New (Correct Local Path):
#LOCAL_PROCESSED_PATH = "/content/Tourism-Package-Prediction/data/processed"

# Update your read_csv calls:
#Xtrain = pd.read_csv(f"{LOCAL_PROCESSED_PATH}/Xtrain.csv")
#Xtest = pd.read_csv(f"{LOCAL_PROCESSED_PATH}/Xtest.csv")
#ytrain = pd.read_csv(f"{LOCAL_PROCESSED_PATH}/ytrain.csv")
#ytest = pd.read_csv(f"{LOCAL_PROCESSED_PATH}/ytest.csv")
# Inside src/train.py
# Use the local path where saved the files

# ----------------------------
# Project Paths
# ----------------------------
BASE_PATH = "/content/Tourism-Package-Prediction"
PROCESSED_DATA_PATH = f"{BASE_PATH}/data/processed"
MODEL_DIR = f"{BASE_PATH}/models"

import os
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_DIR = "/content/Tourism-Package-Prediction/models"
os.makedirs(MODEL_DIR, exist_ok=True)



print(" Training and testing data loaded")

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
# Preprocessing pipeline
# ----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# ----------------------------
# Model & Hyperparameters
# ----------------------------
xgb_model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    objective="reg:squarederror"
)

param_grid = {
    "xgbregressor__n_estimators": [100, 200],
    "xgbregressor__max_depth": [3, 5],
    "xgbregressor__learning_rate": [0.05, 0.1],
    "xgbregressor__subsample": [0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.8, 1.0],
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
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    metrics = {
        "train_RMSE": np.sqrt(mean_squared_error(ytrain, y_pred_train)),
        "test_RMSE": np.sqrt(mean_squared_error(ytest, y_pred_test)),
        "train_MAE": mean_absolute_error(ytrain, y_pred_train),
        "test_MAE": mean_absolute_error(ytest, y_pred_test),
        "train_R2": r2_score(ytrain, y_pred_train),
        "test_R2": r2_score(ytest, y_pred_test),
    }

    mlflow.log_metrics(metrics)

    # ----------------------------
    # Save model locally
    # ----------------------------
    model_path = f"{MODEL_DIR}/tourism_xgb_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print("Model trained and logged in MLflow")

# ----------------------------
# Upload model to Hugging Face
# ----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Sriranjan/Tourism-Package-Pred"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("HF model repo exists")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("HF model repo created")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_xgb_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload trained XGBoost model"
)

print("Model uploaded to Hugging Face")

