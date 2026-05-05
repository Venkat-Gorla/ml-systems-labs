from __future__ import annotations
import numpy as np


def _compute_bins(expected: np.ndarray, bins: int = 10) -> np.ndarray:
    """Compute quantile-based bin edges."""
    return np.percentile(expected, np.linspace(0, 100, bins + 1))


def _compute_distribution(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Compute normalized histogram with epsilon smoothing."""
    counts, _ = np.histogram(values, bins=bins)
    dist = counts / len(values)

    epsilon = 1e-6
    return np.where(dist == 0, epsilon, dist)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute PSI for a single feature."""
    if bins <= 1:
        raise ValueError("bins must be > 1")

    if expected.size == 0 or actual.size == 0:
        raise ValueError("input arrays must be non-empty")

    if np.min(expected) == np.max(expected):
        return 0.0

    bin_edges = _compute_bins(expected, bins)

    expected_dist = _compute_distribution(expected, bin_edges)
    actual_dist = _compute_distribution(actual, bin_edges)

    psi = np.sum((actual_dist - expected_dist) *
                 np.log(actual_dist / expected_dist))
    return float(psi)


def compute_feature_psi(
    X_expected: np.ndarray,
    X_actual: np.ndarray,
    feature_indices: list[int],
) -> dict:
    """Compute PSI per feature and summary stats."""
    if X_expected.shape != X_actual.shape:
        raise ValueError("X_expected and X_actual must have same shape")

    if not feature_indices:
        raise ValueError("feature_indices cannot be empty")

    psi_scores = {}

    for idx in feature_indices:
        psi_scores[idx] = compute_psi(
            X_expected[:, idx],
            X_actual[:, idx]
        )

    values = list(psi_scores.values())

    return {
        "feature_psi": psi_scores,
        "mean_psi": float(np.mean(values)),
        "max_psi": float(np.max(values)),
    }


# ---------- debug / test ----------

def print_psi_result(result: dict) -> None:
    print("Feature PSI:")
    for k, v in result["feature_psi"].items():
        print(f"Feature {k}: {v:.4f}")

    print(f"\nMean PSI: {result['mean_psi']:.4f}")
    print(f"Max PSI:  {result['max_psi']:.4f}")


def test_psi_detects_distribution_shift() -> None:
    np.random.seed(42)

    n_samples = 1000
    n_features = 3
    feature_indices = list(range(n_features))

    X_expected = np.random.normal(0, 1, (n_samples, n_features))
    X_actual = np.random.normal(0.3, 1.2, (n_samples, n_features))

    baseline = compute_feature_psi(X_expected, X_expected, feature_indices)
    result = compute_feature_psi(X_expected, X_actual, feature_indices)

    assert result["mean_psi"] > baseline["mean_psi"] + 0.05, \
        "PSI too low → drift not detected"

    print("\n=== Drift Detection Test ===")
    print_psi_result(result)


if __name__ == "__main__":
    test_psi_detects_distribution_shift()
