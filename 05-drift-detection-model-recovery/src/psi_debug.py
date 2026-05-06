from __future__ import annotations
import numpy as np
from psi import (
    compute_feature_psi,
    _compute_bins,
    _compute_distribution,
    _compute_psi_contrib,
)


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

    print("\n=== PSI Drift Detection Test ===")
    print_psi_result(result)


def _print_psi_bins(
    bins: np.ndarray,
    exp_dist: np.ndarray,
    act_dist: np.ndarray,
):
    for i in range(len(bins) - 1):
        delta = act_dist[i] - exp_dist[i]
        print(
            f"[{bins[i]:>6.2f}, {bins[i+1]:>6.2f}] | "
            f"exp={exp_dist[i]:>7.4f}  "
            f"act={act_dist[i]:>7.4f}  "
            f"delta={delta:>+8.4f}"
        )


def test_psi_bin_distribution_debug():
    np.random.seed(42)

    n = 1000

    expected = np.random.normal(0, 1, n)
    actual = np.random.normal(0.3, 1.2, n)

    bins = _compute_bins(expected, bins=5)
    exp_dist = _compute_distribution(expected, bins)
    act_dist = _compute_distribution(actual, bins)

    psi_contrib = _compute_psi_contrib(exp_dist, act_dist)
    total_psi = np.sum(psi_contrib)

    print("\n=== PSI Bin Distribution Debug Test ===")
    _print_psi_bins(bins, exp_dist, act_dist)
    print(f"\nTotal PSI: {total_psi:.4f}")


if __name__ == "__main__":
    test_psi_detects_distribution_shift()
    test_psi_bin_distribution_debug()
