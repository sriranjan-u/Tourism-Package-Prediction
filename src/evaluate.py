# ============================
# Model Evaluation Script
# ============================
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from huggingface_hub import hf_hub_download
import mlflow

# ----------------------------
# MLflow Configuration
# ----------------------------
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-training")

# ----------------------------
# Hugging Face Dataset Paths
# ----------------------------
HF_DATASET_REPO = "Sriranjan/Tourism-Package-Pred/data/processed"
Xtest = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv"
)
ytest = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/ytest.csv"
).values.ravel()

print("Data loaded successfully from Hugging Face")



# ----------------------------
# Hugging Face Model Download
# ----------------------------
MODEL_REPO = "Sriranjan/Tourism-Package-Pred"
MODEL_FILENAME = "tourism_xgb_model.joblib"

print("Downloading model from Hugging Face...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME,
    repo_type="model"
)

model = load(model_path)

# ----------------------------
# Predictions
# ----------------------------
print("Generating predictions...")
y_pred_proba = model.predict(Xtest)
y_pred = (y_pred_proba >=0.5).astype(int)

# ----------------------------
# Metrics
# ----------------------------
# Calculate classification metrics
acc = accuracy_score(ytest, y_pred)
f1 = f1_score(ytest, y_pred)
prec = precision_score(ytest, y_pred)
rec = recall_score(ytest, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {prec:.4f}")
#print(f"Recall: {rec:.4f}")

# ----------------------------
# Log to MLflow
# ----------------------------
with mlflow.start_run(run_name="model-evaluation"):
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_score", prec)

print("Evaluation completed successfully.")
