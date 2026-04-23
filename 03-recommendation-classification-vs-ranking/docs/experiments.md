# Experiments Log

## Dataset

- file: data/recommendation.csv
- rows: 20
- split: 50/50

## Results

| Date       | Approach       | Accuracy | Precision | Recall | F1    | Pred+ | Key Insight                                     | Commit  |
| ---------- | -------------- | -------- | --------- | ------ | ----- | ----- | ----------------------------------------------- | ------- |
| 2026-04-23 | Classification | 0.500    | 0.333     | 0.250  | 0.286 | 3     | Weak performance with naive features            | b2ee578 |
| 2026-04-23 | Ranking        | -        | 0.333\*   | -      | -     | -     | Weak ranking due to poor feature representation | ff811ca |

\* Precision@2

## Repro

- classification → `uv run src/classification_model.py`
- ranking → `uv run src/ranking_model.py`

## Notes

### Classification

- Initial model predicted only negative class (no positives)
- Adding class_weight="balanced" enabled meaningful predictions
- Performance remains weak due to naive feature representation

### Ranking

- Precision@2 used as evaluation metric
- Users with <2 items excluded from evaluation
- Slight metric improvement after filtering invalid users
- Ranking quality remains weak due to lack of meaningful features

## Key Learning

- Classification and ranking both fail with naive features
- Ranking requires meaningful feature representation to be effective
- Evaluation strategy (e.g., filtering invalid users) impacts metrics
- Model choice is less important than data representation

## Final Conclusion

- Neither classification nor ranking performs well with current features
- Correct problem framing (ranking vs classification) is necessary but not sufficient
- **Feature quality is the primary driver of ML performance**
