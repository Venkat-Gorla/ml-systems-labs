# 05 — Customer Segmentation (Unsupervised Learning)

## Problem

We have customer-level data (age, income, spending) but no clear segmentation.
Without grouping similar customers, targeting and decision-making remain broad and inefficient.

## Goal

Identify natural customer segments and validate them through simple visualization and interpretation.

## Approach

- Select core behavioral features (Age, Income, Spending)
- Standardize features to ensure fair distance calculation
- Apply KMeans clustering (k=5)
- Interpret clusters using original feature values
- Project data to 2D using PCA for visualization

## Tech Stack

| Category          | Tools                 |
| ----------------- | --------------------- |
| **Language**      | Python                |
| **Environment**   | VS Code, Google Colab |
| **Data Handling** | Pandas, NumPy         |
| **Modeling**      | Scikit-learn          |
| **Visualization** | Matplotlib            |

## Results

- Identified 5 distinct customer segments

- Clear separation observed in PCA-based 2D visualization (~78% variance retained)

- Clusters show meaningful behavioral patterns:
  - High income, high spending → premium segment
  - Low income, high spending → impulsive segment
  - High income, low spending → low-engagement segment
  - Moderate groups with balanced behavior

- Cluster assignments persisted for downstream analysis

## Key Takeaways

- Scaling is critical for distance-based models like KMeans
- Clustering can reveal non-obvious behavioral patterns
- PCA is effective for validating cluster separability visually
- Income alone does not explain spending behavior

## Outputs

- CLI pipeline for clustering and interpretation
- Processed dataset with cluster labels
- PCA visualization (2D scatter plot)
- Notebook for exploratory analysis

## Scope

- Short POC (1–2 days)
- Focus on clarity and intuition
- No production considerations
