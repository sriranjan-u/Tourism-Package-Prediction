from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from preprocess import preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    dump(model, "models/tourism_model.pkl")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
