# POC-5 — Detecting Data Drift & Model Degradation

## Problem

ML models degrade silently in production due to changing data patterns.

> **Key question:**
> When should we **trust the model vs retrain or intervene**?

## Goal

- Simulate **data drift scenarios**
- Measure **impact on model performance**
- Evaluate **drift detection strategies**
- Define **action thresholds (when to retrain / alert)**

## Validation Approach

### Baseline

- Assume **static data distribution**
- Train once, evaluate once
- No monitoring

**Why this matters:**
Represents naive production setup → helps quantify hidden risk

### ML / Advanced Approach

- Model: Logistic Regression (simple, interpretable)
- Features: Standard numerical / TF-IDF (depending on dataset)
- Monitoring:
  - Data drift detection (distribution shift)
  - Performance tracking over time

#### Drift Simulation

Introduce controlled shifts:

- Feature distribution change (e.g., mean/variance shift)
- Label distribution change (class imbalance)
- Noise injection

## Comparison Metrics

### Model Performance

- Accuracy
- Precision
- Recall
- F1 Score

### Drift Detection

- Population Stability Index (PSI)
- KL Divergence (optional)
- Feature-wise distribution shift

### System Signals (Important)

- Prediction confidence drop
- Error rate increase over time

## Tech Stack

| Category          | Tools                |
| ----------------- | -------------------- |
| **Language**      | Python               |
| **Data**          | pandas, numpy        |
| **ML**            | scikit-learn         |
| **Evaluation**    | sklearn metrics      |
| **Drift**         | scipy / custom PSI   |
| **Visualization** | matplotlib / seaborn |
| **Cloud**         | (optional) AWS S3    |

## Results Summary

- Performance drops as drift increases (quantified)
- Certain metrics fail to detect early degradation
- Drift signals (PSI etc.) can detect issues earlier than accuracy

## Decision

- Define **trigger thresholds**, e.g.:
  - PSI > 0.2 → investigate
  - F1 drop > X% → retrain

- Combine:
  - **Performance monitoring + drift detection**

- Avoid relying on accuracy alone

## Expected Outcome

> A clear framework for:

- Detecting when models are no longer reliable
- Deciding when to retrain or intervene

## Notes

### POC Focus

- Simulating real-world failure modes
- Understanding monitoring signals
- Designing decision thresholds

### Not Focused On

- Advanced models
- Production pipelines
- Real-time systems

## Reflection (Add after completion)

- What failed first: metrics or drift signals?
- Which signal was most reliable?
- How early could degradation be detected?
- What would you monitor in production?

## 🔥 Small Upgrade (Important)

Add this mindset:

> This POC is not about drift detection accuracy
> It is about **decision confidence under uncertainty**
