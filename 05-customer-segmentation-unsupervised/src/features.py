import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant numeric features for clustering.
    """
    X = df[FEATURE_COLUMNS].copy()

    assert not X.isnull().any().any(), "Null values found in features"

    return X


def scale_features(X: pd.DataFrame) -> np.ndarray:
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # mean and std validation after scaling
    mean = X_scaled.mean(axis=0)
    std = X_scaled.std(axis=0)

    assert np.allclose(mean, 0, atol=1e-7)
    assert np.allclose(std, 1, atol=1e-7)

    print("\n=== Scaling Check ===")
    print("Mean (approx 0):", mean)
    print("Std (approx 1):", std)

    return X_scaled
