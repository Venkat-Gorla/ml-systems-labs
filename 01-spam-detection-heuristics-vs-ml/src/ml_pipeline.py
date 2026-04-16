from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


DATA_PATH = Path("data/spam.csv")
RESULTS_PATH = Path("results/metrics.json")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="latin-1")[["label", "message"]]


def split_data(df: pd.DataFrame):
    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # critical for imbalance
    )


def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def save_results(metrics: dict):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "ml",
        "version": "v1-logistic-regression",
        "metrics": metrics,
    }

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_vec, X_test_vec = vectorize(X_train, X_test)

    model = train_model(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    metrics = evaluate(y_test, y_pred)

    print("ML Results:", metrics)

    save_results(metrics)


if __name__ == "__main__":
    main()
