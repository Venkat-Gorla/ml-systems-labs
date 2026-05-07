from __future__ import annotations

from math import sqrt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": sqrt(mean_squared_error(y_true, y_pred)),
    }


def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
