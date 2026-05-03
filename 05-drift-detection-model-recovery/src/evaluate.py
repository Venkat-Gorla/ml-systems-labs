from __future__ import annotations

from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return precision, recall, f1
