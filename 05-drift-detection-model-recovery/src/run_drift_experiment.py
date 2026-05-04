from __future__ import annotations

import argparse

from data import load_data, split_data
from model import train_model
from drift import apply_feature_drift
from evaluate import evaluate
from experiment_logger import log_experiment


# vegorla: command line parameter for save
# features array usage
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--drift_strength", type=float, default=0.5)
    parser.add_argument("--features", type=int, nargs="+", default=[0, 1])

    return parser.parse_args()


def main():
    args = parse_args()

    # Load + train
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)

    # Apply drift
    X_test_drifted, drift_info = apply_feature_drift(
        X_test,
        drift_strength=args.drift_strength,
        feature_indices=args.features,
    )

    # Evaluate
    precision, recall, f1 = evaluate(model, X_test_drifted, y_test)

    print("\n=== Drift Experiment ===")
    print(f"Drift strength: {args.drift_strength}")
    print(f"Features: {args.features}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Log
    log_experiment(
        drift_type=drift_info["drift_type"],
        drift_strength=drift_info["drift_strength"],
        features=drift_info["features"],
        precision=precision,
        recall=recall,
        f1=f1,
    )


if __name__ == "__main__":
    main()
