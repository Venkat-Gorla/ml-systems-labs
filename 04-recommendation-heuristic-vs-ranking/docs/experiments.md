# Experiments Log

## Dataset

- file: data/interactions.csv
- users: 80
- items: 40
- interactions per user: 6
- split: leave-one-out (1 test item per user)
- negatives: 3 per user (random sampling)

## Results

| Date       | Approach   | Precision@5 | Key Insight                                | Commit  |
| ---------- | ---------- | ----------- | ------------------------------------------ | ------- |
| 2026-04-29 | Heuristic  | 0.0175      | Strong baseline using global popularity    | cb86dc5 |
| 2026-04-29 | ML (Naive) | 0.0150      | Underperforms heuristic with weak features | cb86dc5 |

## Repro

- heuristic → `uv run src/heuristic_model.py`
- ml → `uv run src/ml_model.py`

## Notes

### Heuristic

- Uses global item popularity
- Filters seen items per user
- No training required
- Deterministic output

### ML (Naive)

- Model: LogisticRegression
- Features:
  - user_id
  - item_id
- Negative sampling:
  - 3 negatives per user
- Ranking via predicted probability

### Observations

- ML produces valid ranking (scores + ordering)
- However, ranking quality is poor
- Test items often ranked near bottom (e.g., rank ~34/35)
- Model is confidently wrong due to weak feature representation

## Key Learning

- Heuristic baseline is strong and hard to beat
- Naive ML does not outperform heuristic
- ML requires meaningful features to add value
- Negative sampling introduces noise (false negatives possible)

## Final Conclusion

- ML does NOT justify its complexity at this stage
- Heuristic remains the better choice with current data
- **Feature engineering is required for ML to be competitive**
