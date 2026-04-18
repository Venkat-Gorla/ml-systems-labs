# Threshold Tuning Results (Logistic Regression)

## Setup

- Model: Logistic Regression
- Features: TF-IDF (max_features=5000)
- Dataset Split: 80/20 (stratified)

## Results

| Threshold | Accuracy | Precision | Recall |
| --------- | -------- | --------- | ------ |
| 0.1       | 0.9229   | 0.6413    | 0.9597 |
| 0.2       | 0.9776   | 0.9079    | 0.9262 |
| 0.3       | 0.9758   | 0.9552    | 0.8591 |
| 0.4       | 0.9776   | 0.9921    | 0.8389 |
| 0.5       | 0.9704   | 1.0000    | 0.7785 |
| 0.6       | 0.9534   | 1.0000    | 0.6510 |
| 0.7       | 0.9336   | 1.0000    | 0.5034 |
| 0.8       | 0.9112   | 1.0000    | 0.3356 |
| 0.9       | 0.8771   | 1.0000    | 0.0805 |

## Observations

- Model is highly conservative at default threshold (0.5)
- Precision remains near-perfect across thresholds ≥ 0.4
- Recall drops significantly as threshold increases
- Lower thresholds recover missed spam messages

## Decision

- Selected threshold: **0.2**
- Reason:
  - Strong recall improvement (~0.93)
  - Acceptable precision (~0.91)
  - Best balance for spam detection use case

## Key Insight

- Model quality is not the issue — decision threshold is
- Significant performance gain achieved without retraining
