# Experiments Log

## [2026-04-22] Baseline (All Negative)

### Data

- file: data/fraud.csv
- rows: 40
- split: 50/50

### Model

- baseline_predict → always predicts 0

### Results

- accuracy: 0.5
- precision: 0.0
- recall: 0.0
- f1: 0.0

### Notes

- dataset balanced
- baseline predicts only negative class
- confirms evaluation pipeline works

### Repro

- commit: `988d2cbb4b9621d579f8050a4bda1be8f5d56e66`
- command: `uv run src/heuristic_model.py`

## [2026-04-22] ML Model (Logistic Regression)

### Data

- file: data/fraud.csv
- rows: 40
- split: 50/50

### Model

- Logistic Regression (default parameters)
- threshold: 0.5

### Results

- accuracy: 0.833
- precision: 1.000
- recall: 0.667
- f1: 0.800

### Notes

- model detects fraud with no false positives
- misses some fraud cases (recall < 1.0)
- clear improvement over baseline
- metrics now meaningful (non-zero precision/recall)
- good starting point for threshold tuning

### Repro

- commit: `c641e0c1302bf61d6b5fc2071473fbdae97ac7d6`
- command: `uv run src/ml_model.py`
