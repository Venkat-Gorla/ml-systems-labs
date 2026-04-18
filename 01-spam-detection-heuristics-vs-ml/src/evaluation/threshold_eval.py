import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def run_threshold_tuning(model, X_test_vec, y_test):
    """
    Evaluate model performance across different probability thresholds.
    """
    if isinstance(X_test_vec[0], str):
        raise TypeError("Expected vectorized input, got raw text")

    probs = model.predict_proba(X_test_vec)[:, 1]

    print("\n=== Threshold Tuning ===")

    best_f1 = 0.0
    best_threshold = 0.0

    for threshold in generate_thresholds():
        y_pred = (probs >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)

        f1 = compute_f1(precision, recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        print(
            f"Threshold={threshold:.1f} | "
            f"Acc={accuracy:.4f} | "
            f"Prec={precision:.4f} | "
            f"Rec={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    print("\n=== Best Threshold ===")
    print(f"Threshold={best_threshold:.1f} | F1={best_f1:.4f}")

    return best_threshold, best_f1


def generate_thresholds(start=0.1, stop=0.9, steps=9):
    return np.linspace(start, stop, steps)


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
