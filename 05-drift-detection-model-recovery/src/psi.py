from __future__ import annotations
import numpy as np


def _compute_bins(expected: np.ndarray, bins: int = 10):
    return np.linspace(np.min(expected), np.max(expected), bins + 1)


def _compute_distribution(values: np.ndarray, bins: np.ndarray):
    counts, _ = np.histogram(values, bins=bins)
    dist = counts / len(values)

    # avoid zero for log
    dist = np.where(dist == 0, 1e-6, dist)
    return dist


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    bin_edges = _compute_bins(expected, bins)

    expected_dist = _compute_distribution(expected, bin_edges)
    actual_dist = _compute_distribution(actual, bin_edges)

    psi = np.sum((expected_dist - actual_dist) *
                 np.log(expected_dist / actual_dist))
    return psi


def compute_feature_psi(
    X_expected: np.ndarray,
    X_actual: np.ndarray,
    feature_indices: list[int],
):
    psi_scores = {}

    for idx in feature_indices:
        psi = compute_psi(X_expected[:, idx], X_actual[:, idx])
        psi_scores[idx] = psi

    mean_psi = float(np.mean(list(psi_scores.values())))
    max_psi = float(np.max(list(psi_scores.values())))

    return {
        "feature_psi": psi_scores,
        "mean_psi": mean_psi,
        "max_psi": max_psi,
    }
