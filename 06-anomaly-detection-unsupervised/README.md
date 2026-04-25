# 06 — Anomaly Detection (Unsupervised Learning)

## Problem

In many real-world datasets, rare or unusual observations can have high impact (e.g., fraud, system failures).
These anomalies are typically unlabeled and difficult to detect using rule-based approaches.

## Goal

Identify unusual data points without prior labels and build intuition for how anomaly detection behaves on structured data.

## Approach

- Select relevant numeric features
- Standardize features for fair comparison
- Apply an unsupervised anomaly detection method (e.g., Isolation Forest)
- Score each observation based on “normality”
- Inspect and interpret top anomalies

## Tech Stack

| **Category**      | Tools                 |
| ----------------- | --------------------- |
| **Language**      | Python                |
| **Environment**   | VS Code, Google Colab |
| **Data Handling** | Pandas, NumPy         |
| **Modeling**      | Scikit-learn          |
| **Visualization** | Matplotlib            |

## Results

(To be filled after implementation)

Expected artifacts:

- Anomaly score per observation
- Ranked list of top anomalies
- Visualization highlighting outliers

## Key Takeaways

(To be filled after implementation)

## Outputs

- CLI pipeline for anomaly scoring
- Processed dataset with anomaly flags/scores
- Visualization of normal vs anomalous points

## Scope

- Short POC (1–2 days)
- Focus on intuition and behavior of anomaly detection
- No hyperparameter tuning or production considerations
