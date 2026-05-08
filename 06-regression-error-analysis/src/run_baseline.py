from __future__ import annotations

from data import load_data, split_data
from model import train_model
from evaluate import evaluate, print_metrics
from experiment_logger import log_experiment


def main():
    noise = 10.0
    X, y = load_data(noise=noise)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)

    print("\n=== Regression Baseline ===")
    print_metrics(metrics)

    log_experiment(
        experiment_type="baseline",
        noise=noise,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
    )


if __name__ == "__main__":
    main()
