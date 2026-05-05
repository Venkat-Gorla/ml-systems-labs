from __future__ import annotations

import argparse

from data import load_data, split_data
from model import train_model, get_feature_importance
from drift import apply_feature_drift
from evaluate import evaluate, print_metrics
from experiment_logger import log_experiment


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--drift_strength", type=float, default=0.5)
    parser.add_argument("--features", type=int, nargs="+", default=None)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist evaluation metrics",
    )

    return parser.parse_args()


def resolve_feature_indices(args, model):
    if args.features is None:
        result = get_feature_importance(model)
        top_k = min(3, len(result["ranked_indices"]))
        top_features = result["ranked_indices"][:top_k]
        feature_indices = top_features.tolist()

        print("\nUsing top important features:", feature_indices)
    else:
        feature_indices = args.features
        print("\nUsing user-provided features:", feature_indices)

    return feature_indices


def main():
    args = parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)

    feature_indices = resolve_feature_indices(args, model)
    X_test_drifted, drift_info = apply_feature_drift(
        X_test,
        drift_strength=args.drift_strength,
        feature_indices=feature_indices,
    )

    metrics = evaluate(model, X_test_drifted, y_test)

    print("\n=== Drift Experiment ===")
    print(f"Drift strength: {args.drift_strength}")
    print(f"Features: {feature_indices}")
    print_metrics(metrics)

    if args.save:
        log_experiment(
            drift_type=drift_info["drift_type"],
            drift_strength=drift_info["drift_strength"],
            features=drift_info["features"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )


if __name__ == "__main__":
    main()
