from __future__ import annotations

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def load_data(noise: float = 10):
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        noise=noise,
        random_state=42,
    )

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
