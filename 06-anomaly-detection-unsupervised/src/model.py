import numpy as np
from sklearn.ensemble import IsolationForest


def run_isolation_forest(X_scaled: np.ndarray):
    """
    Train Isolation Forest and return anomaly predictions and scores.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # ~10% anomalies
        random_state=42
    )

    model.fit(X_scaled)

    # Predictions: 1 (normal), -1 (anomaly)
    labels = model.predict(X_scaled)

    # Scores: higher = more normal, lower = more anomalous
    scores = model.decision_function(X_scaled)

    print("\n=== Prediction Sample ===")
    print(labels[:10])

    # Count anomalies
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Prediction Counts ===")
    for u, c in zip(unique, counts):
        label = "Normal" if u == 1 else "Anomaly"
        print(f"{label}: {c}")

    print("\n=== Last 10 Rows (Validation) ===")
    print(labels[-10:])

    print("\nPipeline step complete: anomaly detection done")

    return labels, scores
