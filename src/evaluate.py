# ============================
# Model Evaluation Script
# ============================
import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from huggingface_hub import hf_hub_download
import mlflow

# ----------------------------
# Path & MLflow Configuration
# ----------------------------
# Get the base directory of the repository
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define and set MLflow tracking directory to prevent Permission Errors
MLFLOW_TRACKING_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)

# Set MLflow to use the local workspace
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")
mlflow.set_experiment("tourism-package-training")

# ----------------------------
# Data Loading (Hugging Face Dataset)
# ----------------------------
HF_DATASET_REPO = "Sriranjan/Tourism-Package-Pred/data/processed"

print("Loading test data from Hugging Face...")
Xtest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv")
ytest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytest.csv").values.ravel()

print("Data loaded successfully.")

# ----------------------------
# Model Retrieval
# ----------------------------
MODEL_REPO = "Sriranjan/Tourism-Package-Pred"
MODEL_FILENAME = "models/tourism_xgb_model.joblib" # Note the internal path

print("Downloading model from Hugging Face...")
try:
    # Attempt to download from the repo where the training script uploaded it
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="dataset" # Matches the repo_type in train.py
    )
    model = load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback: check local models folder if HF download fails in runner
    local_path = os.path.join(BASE_DIR, "models", "tourism_xgb_model.joblib")
    if os.path.exists(local_path):
        model = load(local_path)
        print("Loaded model from local storage.")
    else:
        raise FileNotFoundError("Could not find model file on HF or locally.")

# ----------------------------
# Predictions & Metrics
# ----------------------------
print("Generating predictions...")
# Since we are using XGBClassifier, predict() gives class labels directly
y_pred = model.predict(Xtest)

# Calculate classification metrics
acc = accuracy_score(ytest, y_pred)
f1 = f1_score(ytest, y_pred)
prec = precision_score(ytest, y_pred)
rec = recall_score(ytest, y_pred)

print("-" * 30)
print(f"Accuracy:  {acc:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print("-" * 30)

# ----------------------------
# Log to MLflow
# ----------------------------
with mlflow.start_run(run_name="final-model-evaluation"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

print("Evaluation logging completed successfully.")
