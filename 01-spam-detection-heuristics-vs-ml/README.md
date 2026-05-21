# 01 — Spam Detection: Heuristics vs ML

## Problem

Classify incoming messages as **Spam** or **Not Spam**.

The key question is not just _how_ to classify, but:

> **Do we actually need machine learning for this problem?**

## Goal

Evaluate whether a **heuristic (rule-based) approach** is sufficient, or if a **machine learning model** provides meaningful improvement.

This POC focuses on:

- Comparing **heuristics vs ML**
- Understanding **trade-offs (accuracy, recall, complexity)**
- Making a **practical decision**, not just building a model

## Validation Approach

### Heuristic (Baseline)

- Keyword-based rules (e.g., "free", "win", "offer")
- Simple, interpretable, low-cost

### ML Approach

- Logistic Regression with probability-based threshold tuning
- Text features using basic vectorization (TF-IDF or Bag-of-Words)

### Comparison Metrics

- Accuracy
- Precision
- Recall (important for spam detection)
- F1 Score (balanced metric)
- Threshold tuning (precision vs recall trade-off)

## Tech Stack

| Category       | Tools                                             |
| -------------- | ------------------------------------------------- |
| **Language**   | Python                                            |
| **Data**       | pandas                                            |
| **ML**         | scikit-learn (Logistic Regression, TF-IDF)        |
| **Evaluation** | sklearn metrics (accuracy, precision, recall, F1) |
| **Cloud**      | AWS S3, SageMaker                                 |

## Expected Outcome

A clear, reasoned answer to:

> **Is ML justified for this problem, or are heuristics sufficient?**
