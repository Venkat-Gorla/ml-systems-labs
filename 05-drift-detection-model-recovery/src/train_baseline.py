from __future__ import annotations

from data import load_data, split_data
from model import train_model
from evaluate import evaluate


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    precision, recall, f1 = evaluate(model, X_test, y_test)

    print("\n=== Baseline Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    main()
