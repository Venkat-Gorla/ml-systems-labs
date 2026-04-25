# Pipeline Output — Customer Segmentation (KMeans)

## Command

```
uv run src/main.py
```

## Output

```
=== Scaling Check ===
Mean (approx 0): [-1.02140518e-16 -2.13162821e-16 -1.46549439e-16]
Std (approx 1): [1. 1. 1.]

=== Cluster Labels Sample ===
[2 2 3 2 2 2 3 2 0 2]

=== Cluster Counts ===
Cluster 0: 58 points
Cluster 1: 40 points
Cluster 2: 26 points
Cluster 3: 45 points
Cluster 4: 31 points

Pipeline step complete: clustering done

=== Cluster Interpretation (Original Units) ===
Age Annual Income (k$) Spending Score (1-100)
Cluster
0 55.28 47.62 41.71
1 32.88 86.10 81.53
2 25.77 26.12 74.85
3 26.73 54.31 40.91
4 44.39 89.77 18.48

Saved clustered data → data\processed\customers_clustered.csv
```

## Notes

- Features were scaled using `StandardScaler` (mean ≈ 0, std ≈ 1)
- KMeans clustering applied with `k=5`
- Cluster labels align with original dataset row order
- Output dataset persisted for downstream analysis (PCA, visualization)

## Summary

- Data preprocessing and scaling verified
- Clusters generated with reasonable distribution
- Cluster interpretation performed in original feature space
