# POC-5 — Drift Detection & Model Degradation

## Problem

ML models degrade silently as data changes.

> **Key question:**
> When should we trust the model vs retrain?

## Approach

- Train baseline model (Logistic Regression)
- Simulate feature-level drift with varying strength
- Track:
  - Model metrics (Precision, Recall, F1)
  - Data drift using PSI

## Key Results

| Drift Strength | Features | F1     | Mean PSI |
| -------------- | -------- | ------ | -------- |
| 0.0            | []       | 0.8136 | 0.00     |
| 0.5            | [0,1]    | 0.8191 | 0.39     |
| 0.5            | [3,0,5]  | 0.8265 | 0.43     |
| 1.0            | [3,0,5]  | 0.8190 | 1.80     |
| 2.0            | [3,0,5]  | 0.7184 | 6.03     |

## Insights

### 1. Drift does NOT immediately break models

- Metrics remain stable under moderate drift
- Performance may even improve temporarily

### 2. PSI detects issues earlier than metrics

- PSI > 0.25 while F1 is still stable
- Drift is visible before degradation

### 3. Metrics are lagging indicators

- Precision drops gradually
- Recall increases (model bias shift)
- F1 drops only under severe drift

### 4. Important features matter more

- Drift on top features → stronger impact

## Decision Framework

```text
IF mean_psi > 0.25 → investigate
IF mean_psi > 1.0  → high risk
IF mean_psi > 2.0  → retrain
```

Combine with:

- Metric degradation (F1 drop)
- Feature-level drift analysis

## Takeaways

- Monitor **data + model**, not just metrics
- Drift detection is an **early warning system**
- Retraining decisions require **multiple signals**

## Scope

Focused on:

- Drift simulation
- Detection signals
- Decision logic

Not covered:

- Production pipelines
- Real-time monitoring
- Advanced models
