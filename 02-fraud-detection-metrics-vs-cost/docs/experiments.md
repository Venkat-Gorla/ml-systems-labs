# Experiments Log

## Dataset (v1 — clean)

- file: data/fraud.csv
- rows: 40
- split: 50/50

## Dataset (v2 — with overlap / edge cases)

- file: data/fraud.csv
- rows: 45
- split: 50/50
- change: added borderline cases to introduce overlap between fraud and non-fraud

## Results (v1 — baseline vs ML)

| Date       | Model     | Threshold | Accuracy | Precision | Recall | F1    | Key Insight                       | Commit  |
| ---------- | --------- | --------- | -------- | --------- | ------ | ----- | --------------------------------- | ------- |
| 2026-04-22 | Heuristic | -         | 0.500    | 0.000     | 0.000  | 0.000 | Useless (no fraud detected)       | 988d2cb |
| 2026-04-22 | Logistic  | 0.5       | 0.833    | 1.000     | 0.667  | 0.800 | High precision, misses some fraud | c641e0c |

## Results (v2 — threshold experiment)

| Date       | Model    | Threshold | Accuracy | Precision | Recall | F1    | Pred+ | Key Insight                      | Commit  |
| ---------- | -------- | --------- | -------- | --------- | ------ | ----- | ----- | -------------------------------- | ------- |
| 2026-04-22 | Logistic | 0.1       | 0.857    | 0.778     | 1.000  | 0.875 | 9     | Max recall, more false positives | f914bae |
| 2026-04-22 | Logistic | 0.3       | 0.857    | 0.778     | 1.000  | 0.875 | 9     | Same as 0.1 (no boundary change) | f914bae |
| 2026-04-22 | Logistic | 0.5       | 0.786    | 0.750     | 0.857  | 0.800 | 8     | Balanced trade-off               | f914bae |
| 2026-04-22 | Logistic | 0.7       | 0.786    | 0.750     | 0.857  | 0.800 | 8     | Same as 0.5                      | f914bae |

## Repro

- heuristic → `uv run src/heuristic_model.py`
- logistic → `uv run src/ml_model.py`
- threshold → `uv run src/threshold_experiment.py`

## Notes

### Heuristic

- confirms baseline + evaluation pipeline

### Logistic (0.5)

- no false positives (precision = 1.0)
- recall not perfect → misses fraud
- good candidate for threshold tuning

## Key Learning (v2)

- Adding overlap in dataset was necessary to expose trade-offs
- Lower threshold → higher recall, more false positives
- Higher threshold → lower recall, fewer alerts
- Model probabilities are clustered → some thresholds behave identically
- Dataset design directly impacts what ML insights are observable
