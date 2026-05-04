from __future__ import annotations

import argparse
from data import load_data, split_data
from model import train_model
from evaluate import evaluate, print_metrics
from experiment_logger import log_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist evaluation metrics",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    print("\n=== Baseline Metrics ===")
    print_metrics(metrics)

    if args.save:
        log_experiment(
            drift_type="none",
            drift_strength=0.0,
            features=[],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )


if __name__ == "__main__":
    main()
