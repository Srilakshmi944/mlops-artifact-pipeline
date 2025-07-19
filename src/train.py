import json
import joblib
import os
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def load_config(path="config/config.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_data():
    digits = load_digits()
    return digits.data, digits.target

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    config = load_config()
    X, y = load_data()
    model = train_model(X, y, config)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_train.pkl")
    print("✅ Training completed. Model saved to models/model_train.pkl")
