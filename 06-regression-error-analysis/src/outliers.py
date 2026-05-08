from __future__ import annotations

import numpy as np


def inject_outliers(
    y: np.ndarray,
    n_outliers: int = 5,
    multiplier: float = 10.0,
) -> np.ndarray:
    """Inject extreme values into regression targets."""

    if n_outliers <= 0:
        raise ValueError("n_outliers must be > 0")

    if multiplier <= 1:
        raise ValueError("multiplier must be > 1")

    if n_outliers > len(y):
        raise ValueError("n_outliers cannot exceed dataset size")

    y_outlier = y.copy()

    indices = np.random.choice(
        len(y),
        size=n_outliers,
        replace=False,
    )

    y_outlier[indices] *= multiplier

    return y_outlier
