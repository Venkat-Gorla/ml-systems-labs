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

    for threshold in generate_thresholds():
        y_pred = (probs >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)

        print(
            f"Threshold={threshold:.1f} | "
            f"Acc={accuracy:.4f} | "
            f"Prec={precision:.4f} | "
            f"Rec={recall:.4f}"
        )


def generate_thresholds(start=0.1, stop=0.9, steps=9):
    return np.linspace(start, stop, steps)
