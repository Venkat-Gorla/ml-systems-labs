from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
