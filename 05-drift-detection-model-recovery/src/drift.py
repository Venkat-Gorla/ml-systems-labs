from __future__ import annotations

import numpy as np


def apply_feature_drift(
    X: np.ndarray,
    *,
    feature_indices: list[int],
    drift_strength: float = 0.5,
):
    X_drifted = X.copy()

    for idx in feature_indices:
        shift = drift_strength * np.std(X[:, idx])
        X_drifted[:, idx] += shift

    drift_info = {
        "drift_type": "feature_shift",
        "drift_strength": drift_strength,
    }

    return X_drifted, drift_info
