from __future__ import annotations

import argparse
from data import load_data, split_data
from model import train_model
from evaluate import evaluate, print_metrics
from experiment_logger import log_experiment


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--noise",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist experiment metrics",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    X, y = load_data(noise=args.noise)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)

    print("\n=== Regression Baseline ===")
    print(f"Noise: {args.noise}")
    print_metrics(metrics)

    if args.save:
        log_experiment(
            experiment_type="baseline",
            noise=args.noise,
            mae=metrics["mae"],
            rmse=metrics["rmse"],
        )


if __name__ == "__main__":
    main()
