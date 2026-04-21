# 02 — Fraud Detection: Accuracy vs Precision/Recall

## Problem

Detect fraudulent transactions in a **highly imbalanced dataset**, where fraud cases are rare.

Traditional evaluation often uses accuracy, but in this scenario:

- Most transactions are legitimate
- A naive model can achieve high accuracy while being useless

> **Key question:**
> **Which metric should we optimize for — accuracy, precision, or recall — given the cost of fraud vs false alarms?**

## Goal

- Evaluate **accuracy vs precision/recall**
- Understand trade-offs between:
  - Missing fraud (False Negatives)
  - Incorrectly flagging users (False Positives)

- Introduce **cost-based decision-making**
- Optimize:
  - Metric selection
  - Classification threshold

## Validation Approach

### Baseline

- Predict all transactions as **non-fraud**
- (Optional) simple heuristic rules (e.g., high transaction amount)

**Why baseline matters:**

- Exposes the **accuracy illusion** in imbalanced data
- Establishes minimum benchmark

### ML / Advanced Approach

- **Model:** Logistic Regression
- **Features:** Basic numerical features (or preprocessed dataset)
- **Tuning:**
  - Probability threshold (e.g., 0.3, 0.5, 0.7)
  - Observe precision vs recall trade-off

## Comparison Metrics

- Accuracy (to demonstrate failure case)
- Precision (false alarm control)
- Recall (fraud detection capability)
- F1 Score (balance)
- **Cost-based metric (critical):**
  - Assign cost to:
    - False Positives (FP)
    - False Negatives (FN)

  - Compute total system cost

## Tech Stack

| Category            | Tools            |
| ------------------- | ---------------- |
| **Language**        | Python           |
| **Data**            | pandas           |
| **ML**              | scikit-learn     |
| **Evaluation**      | sklearn metrics  |
| **Cloud (planned)** | None (local POC) |

## Results Summary

- Accuracy can be **misleading in imbalanced datasets**
- Precision and Recall expose **real model behavior**
- Threshold tuning significantly impacts outcomes
- **Cost-based evaluation provides the most practical decision signal**

**Final decision:**

- Metric selection must align with **business cost of errors**, not just model performance

## Expected Outcome

> A clear understanding that:
>
> **“The best model is not the one with highest accuracy, but the one that minimizes real-world cost.”**

## Notes

POC focus:

- Fast iteration
- Metric comparison
- Threshold tuning
- Cost-based decision-making

Not focused on:

- Complex models
- Feature engineering depth
- Production deployment
