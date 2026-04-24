import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


def train_kmeans(X_scaled: np.ndarray, k: int = 5) -> KMeans:
    """
    Train KMeans model.
    """
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)

    return model


def inspect_clusters(model: KMeans, X_scaled: np.ndarray) -> None:
    """
    Print basic cluster information.
    """
    labels = model.labels_

    print("\n=== Cluster Labels Sample ===")
    print(labels[:10])

    print("\n=== Cluster Counts ===")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} points")

    print("\n=== Cluster Centers (scaled space) ===")
    print(model.cluster_centers_)

    assert len(
        unique) > 1, "All points assigned to one cluster — something is wrong"


def interpret_clusters(df: pd.DataFrame, labels) -> None:
    """
    Attach cluster labels and show mean feature values per cluster.
    """
    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = labels

    print("\n=== Cluster Interpretation (Original Units) ===")
    summary = df_with_clusters.groupby("Cluster")[[
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]].mean()

    print(summary.round(2))
