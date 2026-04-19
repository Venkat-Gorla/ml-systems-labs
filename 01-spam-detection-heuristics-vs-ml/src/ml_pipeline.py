from pathlib import Path
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.load_and_inspect_data import load_data
from src.cloud.features import split_data, vectorize
from src.save_results import save_metrics
from src.evaluation.confusion_matrix_eval import (
    evaluate_confusion,
    print_misclassified_examples,
)
from src.evaluation.cross_validation_eval import run_cross_validation
from src.evaluation.threshold_eval import run_threshold_tuning

DATA_PATH = Path("data/spam.csv")


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


def main(save: bool, confusion: bool, cv: bool, threshold: bool, df=None):
    if df is None:
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

    if threshold:
        run_threshold_tuning(model, X_test_vec, y_test)

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

    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Run threshold tuning evaluation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        save=args.save,
        confusion=args.confusion,
        cv=args.cv,
        threshold=args.threshold
    )
