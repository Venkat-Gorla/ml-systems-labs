# 06 — Anomaly Detection (Unsupervised Learning)

## Problem

Real-world datasets often contain rare, high-impact observations (e.g., fraud, failures).
These anomalies are unlabeled and cannot be reliably captured using rule-based logic.

## Goal

Detect unusual data points without labels and build intuition for how anomaly detection behaves on structured data.

## Approach

- Standardize features using `StandardScaler`
- Apply `IsolationForest` for anomaly detection
- Generate:
  - Binary labels (`1 = normal`, `-1 = anomaly`)
  - Continuous anomaly scores
- Rank observations by anomaly score
- Visualize anomalies using 2D scatter plot

## Tech Stack

| **Category**      | Tools                 |
| ----------------- | --------------------- |
| **Language**      | Python                |
| **Environment**   | VS Code, Google Colab |
| **Data Handling** | Pandas, NumPy         |
| **Modeling**      | Scikit-learn          |
| **Visualization** | Matplotlib            |

## Results

- Total observations: 60
- Anomalies detected: 6 (~10%, via contamination)

### Key Observations

- Injected high-value points were correctly identified as anomalies
- Not all distant points are anomalous → local density matters
- Model provides both:
  - Hard labels (thresholded)
  - Soft scores (ranking)

## Artifacts

- Processed dataset with anomaly labels + scores
- 2D visualization of anomalies

## Key Takeaways

- Isolation Forest isolates anomalies via random partitioning
- Lower score → more anomalous
- Labels depend on `contamination` (forced threshold)
- Scores provide more reliable insight than labels
- Visualization confirms model behavior effectively

## Scope

- Short POC (1–2 days)
- Focus on clarity and intuition
- No hyperparameter tuning or production optimization

## Next Steps

- Compare with alternative methods (e.g., DBSCAN, LOF)
- Experiment with different contamination values
- Extend to higher-dimensional datasets
