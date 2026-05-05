"""
Simple PSI (Population Stability Index) Implementation

Purpose:
--------
Detects distribution shift between two datasets (expected vs actual).

How it works:
-------------
1. Split the expected data into bins (ranges).
2. Count how many values fall into each bin for:
   - expected data
   - actual data
3. Convert counts into percentages (distributions).
4. Compare the distributions across bins.
5. Larger differences → higher PSI → more drift.

Key idea:
---------
If data distribution changes significantly, PSI increases.

Interpretation:
---------------
PSI < 0.1   → no significant drift
0.1–0.25    → moderate drift
> 0.25      → significant drift (action may be needed)

Notes:
------
- This is a simplified, readable version for learning.
- See psi.py for a more robust, production-style implementation.
"""
from __future__ import annotations

import numpy as np


def _create_bins(values: np.ndarray, num_bins: int = 5):
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    return np.linspace(min_val, max_val, num_bins + 1)


def _get_distribution(values: np.ndarray, bins: np.ndarray):
    counts = np.zeros(len(bins) - 1)

    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break

    # include right edge
    counts[-1] += np.sum(values == bins[-1])

    total = len(values)
    dist = counts / total

    # avoid zeros (simple safeguard)
    dist = np.where(dist == 0, 1e-6, dist)

    return dist


def _compute_simple_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 5) -> float:
    bin_edges = _create_bins(expected, bins)

    expected_dist = _get_distribution(expected, bin_edges)
    actual_dist = _get_distribution(actual, bin_edges)

    psi = 0.0
    for e, a in zip(expected_dist, actual_dist):
        psi += (e - a) * np.log(e / a)

    return float(psi)


def compute_simple_feature_psi(
    X_expected: np.ndarray,
    X_actual: np.ndarray,
    feature_indices: list[int],
):
    psi_scores = {}

    for idx in feature_indices:
        psi = _compute_simple_psi(X_expected[:, idx], X_actual[:, idx])
        psi_scores[idx] = psi

    mean_psi = float(np.mean(list(psi_scores.values())))
    max_psi = float(np.max(list(psi_scores.values())))

    return {
        "feature_psi": psi_scores,
        "mean_psi": mean_psi,
        "max_psi": max_psi,
    }


def _debug_driver():
    import numpy as np

    # small, readable dataset
    X_expected = np.array([
        [10, 100, 0],
        [12, 105, 1],
        [11, 98,  0],
        [13, 102, 1],
        [12, 101, 0],
    ])

    # introduce drift in feature 0 and 1
    X_actual = np.array([
        [20, 120, 0],
        [22, 125, 1],
        [21, 118, 0],
        [23, 122, 1],
        [22, 121, 0],
    ])

    feature_indices = [0, 1, 2]

    result = compute_simple_feature_psi(
        X_expected,
        X_actual,
        feature_indices,
    )

    print("\n=== Simple PSI Debug ===")
    print("Feature PSI:")
    for k, v in result["feature_psi"].items():
        print(f"Feature {k}: {v:.4f}")

    print(f"\nMean PSI: {result['mean_psi']:.4f}")
    print(f"Max PSI:  {result['max_psi']:.4f}")


if __name__ == "__main__":
    _debug_driver()
