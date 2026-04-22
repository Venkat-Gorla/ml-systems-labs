# 03 — Recommendation System: Classification vs Ranking

## Problem

Recommend relevant items (e.g., products, content) to users.

Each user can interact with multiple items, but we must decide:

> **Key question:**
> Should we predict individual interactions (classification), or rank items relative to each other (ranking)?

## Goal

- Compare **classification vs ranking approaches**
- Understand differences in:
  - Problem formulation
  - Output behavior
  - Evaluation strategy
- Identify which approach better supports **top-N recommendations**

## Validation Approach

### Approach A — Classification

- Predict:
  - Will user interact with item? (yes/no)

- Output:
  - Binary prediction per (user, item)

- Usage:
  - Filter items with positive prediction

**Limitations:**

- Does not naturally order items
- Difficult to select top-N recommendations

### Approach B — Ranking

- Predict:
  - Relevance score for each (user, item)

- Output:
  - Continuous score

- Usage:
  - Sort items by score
  - Recommend top-N items

**Advantage:**

- Directly optimized for recommendation scenarios

## Dataset

- Simulated or small dataset:
  - user_id
  - item_id
  - interaction (0/1)

- Each user has multiple candidate items

## Comparison Metrics

### Classification

- Accuracy
- Precision
- Recall
- F1 Score

### Ranking

- Precision@K
- Recall@K
- (Optional) Mean Reciprocal Rank (MRR)

## Evaluation Strategy

- For each user:
  - Generate predictions for multiple items

- Compare:
  - Classification → which items predicted positive
  - Ranking → top-K items by score

- Evaluate:
  - How many relevant items appear in top-K

## Results Summary (Expected)

- Classification works for:
  - Binary decisions
  - Simple filtering

- Ranking works better for:
  - Ordering items
  - Top-N recommendation

## Expected Outcome

> A clear understanding that:

**“Recommendation problems are better framed as ranking problems, not classification.”**

## Notes

POC focus:

- Problem framing
- Output interpretation
- Evaluation differences

Not focused on:

- Advanced recommendation algorithms
- Feature engineering
- Production-scale systems
