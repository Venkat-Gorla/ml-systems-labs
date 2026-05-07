import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["Feature1", "Feature2"]


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """
    Scale features using StandardScaler.
    """
    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # mean and std validation after scaling
    mean = X_scaled.mean(axis=0)
    std = X_scaled.std(axis=0)

    assert np.allclose(mean, 0, atol=1e-7), "Mean not ~0 after scaling"
    assert np.allclose(std, 1, atol=1e-7), "Std not ~1 after scaling"

    print("\n=== Scaling Check ===")
    print("Mean (approx 0):", mean)
    print("Std (approx 1):", std)

    return X_scaled
