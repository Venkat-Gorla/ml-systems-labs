from __future__ import annotations

import numpy as np


def inject_outliers(
    y: np.ndarray,
    n_outliers: int = 5,
    multiplier: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject extreme values into regression targets."""

    if n_outliers <= 0:
        raise ValueError("n_outliers must be > 0")

    if multiplier <= 1:
        raise ValueError("multiplier must be > 1")

    if n_outliers > len(y):
        raise ValueError("n_outliers cannot exceed dataset size")

    y_outlier = y.copy()

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(y),
        size=n_outliers,
        replace=False,
    )

    y_outlier[indices] *= multiplier

    return y_outlier, indices


def _debug_driver():
    y = np.array([10, 20, 30, 40, 50], dtype=float)

    y_outlier, indices = inject_outliers(
        y,
        n_outliers=2,
        multiplier=10,
    )

    print("Original:", y)
    print("Outlier :", y_outlier)
    print("Indices :", indices)


if __name__ == "__main__":
    _debug_driver()
