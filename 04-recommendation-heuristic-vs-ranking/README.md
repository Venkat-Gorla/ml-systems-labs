# 04 — Recommendation: Heuristic vs ML

## Problem

Recommend relevant items to users based on past interactions.

Unlike previous POC (classification vs ranking), the focus here is:

> **Does machine learning provide meaningful improvement over simple heuristics?**

## Goal

- Compare **heuristic recommender vs ML-based recommender**
- Evaluate trade-offs:
  - Performance (Precision@K)
  - Complexity
  - Interpretability
- Understand when ML is justified

## Validation Approach

### Heuristic Baseline

- Simple recommendation strategy:
  - Global popularity (most interacted items)
  - (Optional) user-based rules

**Why this matters:**

- Strong baseline in many real systems
- Zero training cost
- Easy to explain

### ML Approach (Ranking Model)

ML in this POC refers to a **ranking model** that assigns scores to items
and generates recommendations via sorting (top-K selection).

#### Stage 1 — Naive

- **Model:** Logistic Regression
- **Features:**
  - user_id
  - item_id

**Purpose:**

- Establish weak ML baseline
- Compare fairly with heuristic using same limited information

#### Stage 2 — Improved Features

- Add simple features:
  - item popularity (interaction count)
  - user activity (interaction count)

**Purpose:**

- Observe how feature quality impacts performance
- Measure incremental improvement

## Comparison Metrics

- Precision@K (primary)
- (Optional) Recall@K
- Number of recommendations per user

## Tech Stack

| Category       | Tools            |
| -------------- | ---------------- |
| **Language**   | Python           |
| **Data**       | pandas           |
| **ML**         | scikit-learn     |
| **Evaluation** | custom metrics   |
| **Cloud**      | None (local POC) |

## Results Summary (Expected)

- Heuristic provides strong baseline performance
- Naive ML underperforms due to weak features
- Improved ML may surpass heuristic
- Feature quality drives ML success

## Expected Outcome

> A clear answer to:

**“When does ML justify its complexity over simple heuristics?”**

## Notes

POC focus:

- Fair comparison
- Incremental feature improvement
- Decision-making based on results

Not focused on:

- Advanced recommender systems
- Deep learning / embeddings
- Production-scale optimization
