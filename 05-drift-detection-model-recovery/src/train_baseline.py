from __future__ import annotations

from data import load_data, split_data
from model import train_model
from evaluate import evaluate
from experiment_logger import log_experiment


# vegorla: consider different filename
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    precision, recall, f1 = evaluate(model, X_test, y_test)

    print("\n=== Baseline Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # vegorla: save should be command line parameter
    log_experiment(
        drift_type="none",
        drift_strength=0.0,
        features=[],
        precision=precision,
        recall=recall,
        f1=f1,
    )


if __name__ == "__main__":
    main()
