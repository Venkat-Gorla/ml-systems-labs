# Experiments Log

## Dataset

- file: data/fraud.csv
- rows: 40
- split: 50/50

## Summary

| Date       | Model     | Threshold | Accuracy | Precision | Recall | F1    | Key Insight                       | Commit  |
| ---------- | --------- | --------- | -------- | --------- | ------ | ----- | --------------------------------- | ------- |
| 2026-04-22 | Heuristic | -         | 0.500    | 0.000     | 0.000  | 0.000 | Useless (no fraud detected)       | 988d2cb |
| 2026-04-22 | Logistic  | 0.5       | 0.833    | 1.000     | 0.667  | 0.800 | High precision, misses some fraud | c641e0c |

---

## Repro

- heuristic → `uv run src/heuristic_model.py`
- logistic → `uv run src/ml_model.py`

## Notes

### Heuristic

- confirms baseline + evaluation pipeline

### Logistic (0.5)

- no false positives (precision = 1.0)
- recall not perfect → misses fraud
- good candidate for threshold tuning
