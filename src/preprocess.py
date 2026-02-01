# ============================
# Data Preparation Script
# ============================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Paths (YOUR PROJECT PATH)
# ----------------------------
BASE_PATH = "/content/Tourism-Package-Prediction"
RAW_DATA_PATH = f"{BASE_PATH}/data/raw/tourism.csv"
PROCESSED_DATA_PATH = f"{BASE_PATH}/data/processed"

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv(RAW_DATA_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier (not useful for modeling)
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# ----------------------------
# Encode Categorical Columns
# ----------------------------
categorical_cols = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "ProductPitched", "CityTier", "Passport", "OwnCar", "Designation"
]

label_encoder = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# ----------------------------
# Train-Test Split
# ----------------------------
target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Save Processed Data
# ----------------------------
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

Xtrain.to_csv(f"{PROCESSED_DATA_PATH}/Xtrain.csv", index=False)
Xtest.to_csv(f"{PROCESSED_DATA_PATH}/Xtest.csv", index=False)
ytrain.to_csv(f"{PROCESSED_DATA_PATH}/ytrain.csv", index=False)
ytest.to_csv(f"{PROCESSED_DATA_PATH}/ytest.csv", index=False)

print("Train-test datasets saved to data/processed/")

# OPTIONAL: Upload processed data to Hugging Face Dataset
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Sriranjan/Tourism-Package-Pred"
repo_type = "dataset"

files_to_upload = [
    f"{PROCESSED_DATA_PATH}/Xtrain.csv",
    f"{PROCESSED_DATA_PATH}/Xtest.csv",
    f"{PROCESSED_DATA_PATH}/ytrain.csv",
    f"{PROCESSED_DATA_PATH}/ytest.csv",
]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"data/processed/{os.path.basename(file_path)}",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Upload {os.path.basename(file_path)}"
    )

print("Processed training data uploaded to Hugging Face Dataset")
