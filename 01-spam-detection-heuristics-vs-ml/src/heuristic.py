import re
from typing import List, Tuple
import pandas as pd


DEFAULT_THRESHOLD = 2

SPAM_KEYWORDS: List[str] = [
    "free", "win", "winner", "offer", "click", "buy",
    "urgent", "cash", "prize", "claim", "limited", "now"
]


def compute_score(message: str) -> int:
    text = message.lower()
    score = 0

    # keyword match
    if any(keyword in text for keyword in SPAM_KEYWORDS):
        score += 1

    # link detection
    if "http" in text or "www" in text:
        score += 1

    # numbers (e.g. prizes, phone numbers)
    if len(re.findall(r"\d+", text)) >= 2:
        score += 1

    # ALL CAPS words
    if any(word.isupper() and len(word) > 3 for word in message.split()):
        score += 1

    return score


def is_spam(message: str, threshold: int) -> str:
    return "spam" if compute_score(message) >= threshold else "ham"


def predict(df: pd.DataFrame, threshold: int = DEFAULT_THRESHOLD) -> List[str]:
    return [is_spam(msg, threshold) for msg in df["message"]]


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
