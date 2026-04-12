from typing import List, Tuple
import pandas as pd


SPAM_KEYWORDS: List[str] = [
    "free", "win", "winner", "offer", "click", "buy",
    "urgent", "cash", "prize", "claim", "limited", "now"
]


def is_spam(message: str) -> str:
    text = message.lower()
    return "spam" if any(keyword in text for keyword in SPAM_KEYWORDS) else "ham"


def predict(df: pd.DataFrame) -> List[str]:
    return [is_spam(msg) for msg in df["message"]]


def evaluate(y_true: List[str], y_pred: List[str]) -> Tuple[int, int, int, int]:
    tp = fp = fn = tn = 0

    for true, pred in zip(y_true, y_pred):
        if true == "spam" and pred == "spam":
            tp += 1
        elif true == "ham" and pred == "spam":
            fp += 1
        elif true == "spam" and pred == "ham":
            fn += 1
        else:
            tn += 1

    return tp, fp, fn, tn


def print_metrics(tp: int, fp: int, fn: int, tn: int) -> None:
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print("\n=== Heuristic Model Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


def run(df: pd.DataFrame) -> None:
    y_true = df["label"].tolist()
    y_pred = predict(df)

    tp, fp, fn, tn = evaluate(y_true, y_pred)
    print_metrics(tp, fp, fn, tn)
