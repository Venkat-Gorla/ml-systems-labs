# Experiments Log

## Dataset

- file: data/interactions.csv (V1)
- file: data/interactions_v2.csv (V2)
- users: 80
- items: 40
- interactions per user: 6
- V1 structure:
  - users randomly assigned to 4 groups
  - each group prefers a subset of items
  - 20% noise (cross-group interactions)
- V2 structure:
  - deterministic user-item groups (A/B)
  - users interact mostly within their group (5 preferred + 1 noise)
- split: leave-one-out (1 test item per user)
- negatives:
  - V1: 3 random unseen items per user (mixed signal)
  - V2: 3 non-preferred items per user (group-aware)

## Results

| Date       | Approach      | Precision@5 | Key Insight                                                   | Commit  |
| ---------- | ------------- | ----------- | ------------------------------------------------------------- | ------- |
| 2026-04-29 | Heuristic     | 0.0175      | Strong baseline using global popularity                       | cb86dc5 |
| 2026-04-29 | ML (V1)       | 0.0150      | Fails due to hidden structure + weak features + noise         | cb86dc5 |
| 2026-05-01 | ML (V2)       | 0.0300      | Structured data + aligned sampling → ML outperforms heuristic | 13f253a |
| 2026-05-02 | ML (V2 + pop) | 0.0325      | Item popularity provides small but consistent lift            | 297575a |

## Repro

- heuristic → `uv run src/heuristic_model.py`
- ml (V1) → `uv run src/ml_model.py`
- ml (V2) → `uv run src/ml_model_v2.py`
- ml (V2 + pop) → `uv run src/ml_model_v2.py --use-popularity`

## Notes

### Heuristic

- Filters seen items per user
- Deterministic

### ML (V1)

- Model: LogisticRegression
- Features:
  - user_id
  - item_id
- Ranking via predicted probability

## Observations

- V1 Dataset contains **implicit group structure**, but:
  - user groups are randomly assigned and not encoded in features
  - model cannot directly infer group membership
- Negative sampling mixes preferred and non-preferred items → weak learning signal
- Test items often ranked near bottom (~34/35)

## Key Learning

- Not all structure is learnable:
  - signal must align with observable features
- Heuristic remains the better choice with current setup
- **Data representation and feature alignment are critical for ML success**
