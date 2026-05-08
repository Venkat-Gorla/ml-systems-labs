# POC-6 — Regression Error Analysis

## Problem

Regression models predict continuous values.

> **Key question:**
> How do regression error metrics behave under different failure scenarios?

## Goal

- Build a simple regression pipeline
- Understand regression error metrics:
  - **MAE**
  - **RMSE**
- Analyze metric behavior under:
  - increasing noise
  - extreme outliers

This POC focuses on regression metric behavior under controlled experimental conditions.

## Approach

- Generate synthetic regression data
- Train Linear Regression model
- Evaluate predictions using:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

## Experiments

### 1. Baseline

Train and evaluate regression model on normal data.

Baseline result:

| Noise |    MAE |    RMSE |
| ----- | -----: | ------: |
| 10    | 8.5351 | 10.5312 |

### 2. Noise Increase

Increase dataset noise and observe metric degradation.

| Noise |     MAE |    RMSE |
| ----- | ------: | ------: |
| 10    |  8.5351 | 10.5312 |
| 30    | 25.6053 | 31.5936 |
| 60    | 51.2106 | 63.1872 |

Observation:

- Both MAE and RMSE increased steadily as noise increased
- **RMSE** consistently increased more aggressively

### 3. Outlier Injection

Inject extreme target values into evaluation data.

| Experiment        |     MAE |    RMSE |
| ----------------- | ------: | ------: |
| Baseline          |  8.5351 | 10.5312 |
| Outlier Injection | 15.9425 | 57.7480 |

Observation:

- MAE increased moderately
- RMSE increased dramatically
- **RMSE** is highly sensitive to rare extreme prediction failures

## Key Learnings

- **MAE** measures average prediction error linearly
- **RMSE** penalizes large errors more aggressively
- Noise causes gradual degradation
- Outliers disproportionately affect RMSE

## Conclusion

This POC demonstrated how regression metrics behave under controlled failure scenarios.

The most important insight was that **RMSE reacts much more strongly to extreme prediction errors than MAE**, making it useful for detecting catastrophic regression failures.
