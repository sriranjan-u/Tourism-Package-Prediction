import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/raw/tourism.csv"

def preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Handle missing values
    df = df.dropna()

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
