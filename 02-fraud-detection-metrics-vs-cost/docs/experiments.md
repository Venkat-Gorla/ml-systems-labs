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
