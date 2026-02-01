import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Configuration
REPO_ID = "Sriranjan/Tourism-Package-Pred"
RAW_DATA_PATH = "hf://datasets/Sriranjan/Tourism-Package-Pred/data/raw/tourism.csv"
LOCAL_DATA_DIR = "/content/Tourism-Package-Prediction/data/processed"

def preprocess():
    print("Loading raw data from Hugging Face...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Cleaning: Drop CustomerID and handle missing values
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)
    df.fillna(method='ffill', inplace=True)

    # 2. Encoding: Transform categorical columns
    categorical_cols = [
        "TypeofContact", "Occupation", "Gender", "MaritalStatus",
        "ProductPitched", "CityTier", "Passport", "OwnCar", "Designation"
    ]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # 3. Splitting: 80/20 split for ProdTaken
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Save Locally
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    X_train.to_csv(f"{LOCAL_DATA_DIR}/Xtrain.csv", index=False)
    X_test.to_csv(f"{LOCAL_DATA_DIR}/Xtest.csv", index=False)
    y_train.to_csv(f"{LOCAL_DATA_DIR}/ytrain.csv", index=False)
    y_test.to_csv(f"{LOCAL_DATA_DIR}/ytest.csv", index=False)
    print(f"Data saved locally to {LOCAL_DATA_DIR}")

if __name__ == "__main__":
    preprocess()
