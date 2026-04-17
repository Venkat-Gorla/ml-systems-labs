from pathlib import Path
import argparse

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from load_and_inspect_data import load_data
from save_results import save_metrics
from evaluation.confusion_matrix_eval import (
    evaluate_confusion,
    print_misclassified_examples,
)
from evaluation.cross_validation_eval import run_cross_validation

DATA_PATH = Path("data/spam.csv")


def split_data(df: pd.DataFrame):
    X = df["message"]
    y = df["label"].str.lower().map({"ham": 0, "spam": 1})

    if y.isnull().any():
        raise ValueError("Unknown labels found during mapping")

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # keep same spam/ham ratio in train & test
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


def round_metrics(metrics: dict, decimals: int = 4) -> dict:
    return {k: round(v, decimals) for k, v in metrics.items()}


def main(save: bool, confusion: bool, cv: bool):
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_vec, X_test_vec = vectorize(X_train, X_test)

    model = train_model(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = evaluate(y_test, y_pred)
    metrics = round_metrics(metrics)

    print("=== ML Model Metrics ===")
    print(metrics)

    if confusion:
        evaluate_confusion(y_test, y_pred)
        print_misclassified_examples(X_test, y_test, y_pred)

    if cv:
        y_full = df["label"].str.lower().map({"ham": 0, "spam": 1})
        run_cross_validation(df["message"], y_full)

    if save:
        save_metrics(
            metrics,
            model_name="ml",
            version="v1-logistic-regression"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="ML Spam Detection Pipeline")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist evaluation metrics",
    )

    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Run confusion matrix evaluation",
    )

    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run cross-validation evaluation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(save=args.save, confusion=args.confusion, cv=args.cv)
