from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_confusion(y_true, y_pred) -> Dict[str, int]:
    """
    Compute and print confusion matrix with TN, FP, FN, TP.
    """

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        raise ValueError("Only binary classification is supported.")

    tn, fp, fn, tp = cm.ravel()

    print("\n=== Confusion Matrix ===")
    print(cm)
    print("\nBreakdown:")
    print(f"TN (ham correctly classified): {tn}")
    print(f"FP (ham misclassified as spam): {fp}")
    print(f"FN (spam misclassified as ham): {fn}")
    print(f"TP (spam correctly classified): {tp}")

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_misclassified_examples(
    X,
    y_true,
    y_pred,
    limit: int = 5,
) -> None:
    """
    Print sample false positives and false negatives.
    """

    X = np.array(X)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mis_idx = np.where(y_true != y_pred)[0]

    fp_idx = [i for i in mis_idx if y_true[i] == 0 and y_pred[i] == 1]
    fn_idx = [i for i in mis_idx if y_true[i] == 1 and y_pred[i] == 0]

    print("\n=== False Positives (ham → spam) ===")
    for i in fp_idx[:limit]:
        print(f"- {X[i]}")

    print("\n=== False Negatives (spam → ham) ===")
    for i in fn_idx[:limit]:
        print(f"- {X[i]}")
