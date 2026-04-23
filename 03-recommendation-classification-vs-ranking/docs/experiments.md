# Experiments Log

## Dataset

- file: data/recommendation.csv
- rows: 20
- split: 50/50

## Results

| Date       | Approach       | Accuracy | Precision | Recall | F1    | Pred+ | Key Insight                          | Commit  |
| ---------- | -------------- | -------- | --------- | ------ | ----- | ----- | ------------------------------------ | ------- |
| 2026-04-23 | Classification | 0.500    | 0.333     | 0.250  | 0.286 | 3     | Weak performance with naive features | b2ee578 |

## Repro

- classification → `uv run src/classification_model.py`

## Notes

### Classification

- Initial model predicted only negative class (no positives)
- Adding class_weight="balanced" enabled meaningful predictions
- Performance remains weak due to naive feature representation
