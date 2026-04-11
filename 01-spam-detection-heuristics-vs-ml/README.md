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

- Lightweight classification model (e.g., Logistic Regression)
- Text features using basic vectorization (TF-IDF or Bag-of-Words)

### Comparison Metrics

- Accuracy
- Precision
- Recall (important for spam detection)

## Tech Stack

- **Python**
  - pandas (data handling)
  - scikit-learn (modeling + evaluation)

- **AWS (light usage)**
  - Amazon S3 — dataset storage
  - Amazon SageMaker — model training

## Expected Outcome

A clear, reasoned answer to:

> **Is ML justified for this problem, or are heuristics sufficient?**

## Notes

This is a **POC (Proof of Concept)** focused on:

- Fast iteration
- Decision-making
- Learning through comparison

Not focused on:

- Model optimization
- Production readiness
