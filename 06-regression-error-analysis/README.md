# POC-6 — Regression Error Analysis

## Problem

Regression models predict continuous values.

> **Key question:**
> How do regression error metrics behave under different conditions?

## Goal (MVP)

- Build a simple regression model
- Understand basic error metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

## Approach

- Generate synthetic regression data
- Train Linear Regression model
- Evaluate predictions using MAE and RMSE

## Metrics

- **MAE** → average absolute error
- **RMSE** → penalizes large errors more

## Expected Outcome

- Working regression pipeline
- Clear understanding of:
  - What prediction error means
  - Difference between MAE and RMSE
  - How metrics behave differently under various scenarios

## Scope (MVP)

Focused on:

- Basic regression pipeline
- Metric understanding

Not covered:

- Advanced models
- Feature engineering
- Real-world datasets

## Next Steps (Later)

- Add outliers → compare MAE vs RMSE
- Add noise → observe stability
- Introduce distribution shift
