from joblib import load
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data

def evaluate_model():
    X_train, X_test, y_train, y_test = preprocess_data()

    model = load("models/tourism_model.pkl")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
