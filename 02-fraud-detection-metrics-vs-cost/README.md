# 02 — Fraud Detection: Accuracy vs Precision/Recall

## Problem

Detect fraudulent transactions where:

- Missing fraud (FN) is costly
- False alarms (FP) are undesirable but cheaper

> **Key question:**
> Which metric (accuracy, precision, recall) leads to the best decision given error costs?

## Goal

- Compare accuracy vs precision/recall
- Understand FP vs FN trade-offs
- Evaluate impact of threshold tuning
- Select optimal decision using cost-based evaluation

## Validation Approach

### Baseline

- Predict all transactions as **non-fraud**

**Purpose:**

- Demonstrate accuracy can be misleading

### ML Approach

- **Model:** Logistic Regression
- **Features:** Basic numerical features
- **Tuning:**
  - Threshold (0.1–0.7)
  - Observe precision vs recall trade-off

## Comparison Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Cost (FP vs FN)

## Tech Stack

| Category       | Tools            |
| -------------- | ---------------- |
| **Language**   | Python           |
| **Data**       | pandas           |
| **ML**         | scikit-learn     |
| **Evaluation** | sklearn metrics  |
| **Cloud**      | None (local POC) |

## Results Summary

- Accuracy can be misleading
- Precision and recall reveal trade-offs
- Threshold tuning changes model behavior
- Cost-based evaluation drives the final decision

**Conclusion:**
Metric selection must align with business cost, not accuracy

## Expected Outcome

> A clear understanding that:
>
> **“The best model is not the one with highest accuracy, but the one that minimizes real-world cost.”**

## Notes

POC focus:

- Fast iteration
- Metric comparison
- Threshold tuning
- Cost-based decisions

Not covered:

- Complex models
- Feature engineering
- Production systems
