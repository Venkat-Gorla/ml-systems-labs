# Experiments Log

## Dataset

- file: data/interactions.csv
- users: 80
- items: 40
- interactions per user: 6
- structure:
  - users randomly assigned to 4 groups
  - each group prefers a subset of items
  - 20% noise (cross-group interactions)

- split: leave-one-out (1 test item per user)
- negatives: 3 random unseen items per user (preferred and non-preferred)

## Results

| Date       | Approach   | Precision@5 | Key Insight                              | Commit  |
| ---------- | ---------- | ----------- | ---------------------------------------- | ------- |
| 2026-04-29 | Heuristic  | 0.0175      | Strong baseline using global popularity  | cb86dc5 |
| 2026-04-29 | ML (Naive) | 0.0150      | Fails due to hidden user group structure | cb86dc5 |

## Repro

- heuristic → `uv run src/heuristic_model.py`
- ml → `uv run src/ml_model.py`

## Notes

### Heuristic

- Filters seen items per user
- Deterministic

### ML (Naive)

- Model: LogisticRegression
- Features:
  - user_id
  - item_id

- Ranking via predicted probability

## Observations

- Dataset contains **implicit group structure**, but:
  - user groups are randomly assigned (not encoded in features)
  - model cannot directly infer group membership
- Test items often ranked near bottom (~34/35)

## Key Learning

- Not all structure is learnable:
  - signal must align with observable features

- Heuristic remains the better choice with current setup
- **Data representation and feature alignment are critical for ML success**
