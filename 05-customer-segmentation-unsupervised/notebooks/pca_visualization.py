# -*- coding: utf-8 -*-
"""pca-visualization

# Customer Segmentation — PCA Visualization

## Overview
Visualizes customer segments identified via KMeans by projecting features into 2D using PCA.

## Objective
- Reduce 3D feature space (Age, Income, Spending) to 2D
- Visually validate cluster separation

## Data
- Input: `customers_clustered.csv`
- Features: Age, Annual Income, Spending Score
- Labels: Cluster (from KMeans)

## Output
- 2D scatter plot (PC1 vs PC2)
- Points colored by cluster

## Notes
- PCA is used only for visualization (not clustering)
- Cluster labels are not used as input features
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv("customers_clustered.csv")

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Mean:", X_scaled.mean(axis=0))
print("Std:", X_scaled.std(axis=0))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA shape:", X_pca.shape)
print("First 5 rows:")
print(X_pca[:5])

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance captured:", pca.explained_variance_ratio_.sum())

plt.figure()

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"]
)
plt.colorbar(label="Cluster")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segments (PCA Projection)")

plt.savefig("customer_segments_pca.png", dpi=300, bbox_inches="tight")

plt.show()
