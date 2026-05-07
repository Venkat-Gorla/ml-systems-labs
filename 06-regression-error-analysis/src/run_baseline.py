from __future__ import annotations

from data import load_data, split_data
from model import train_model
from evaluate import evaluate, print_metrics


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)

    print("\n=== Regression Baseline ===")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
