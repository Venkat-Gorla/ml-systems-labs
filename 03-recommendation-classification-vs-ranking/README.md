# 03 — Recommendation System: Classification vs Ranking

## Problem

Recommend relevant items to users.

> **Key question:**
> Should we predict interactions (classification) or rank items (ranking) for top-N recommendations?

## Approach

### Classification

- Predict: interaction (yes/no) per (user, item)
- Output: binary labels → filter positives

**Limitation:**

- No natural ordering → weak for top-N recommendation

### Ranking

- Predict: relevance score per (user, item)
- Output: sorted items → top-N selection

**Advantage:**

- Directly models recommendation use-case

## Metrics

- Classification: Accuracy, Precision, Recall, F1
- Ranking: Precision@K (primary), Recall@K

## Evaluation

- For each user:
  - Score multiple items
  - Compare:
    - classification → predicted positives
    - ranking → top-K items

## Results Summary

- Classification performed poorly with naive features
- Ranking also weak despite correct formulation
- Ranking required filtering users with insufficient test data

**Limitations:**

- Classification → no ordering + weak signals
- Ranking → highly dependent on feature quality

## Key Learning

- Ranking is the correct formulation for recommendation
- But **problem framing alone is not enough**
- **Feature representation is the primary driver of performance**

## Conclusion

> **Recommendation systems should use ranking, but success depends on meaningful features.**

## Notes

Focus:

- Problem framing
- Evaluation differences

Not covered:

- Feature engineering
- Advanced models
- Production systems
