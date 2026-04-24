"""
Quick experiment to understand standardization (mean=0, std=1)
"""
import numpy as np

X = np.array([10, 20, 30])
mean = X.mean()
std = X.std()

X_scaled = (X - mean) / std

print("Scaled:", X_scaled)
print("Mean:", X_scaled.mean())
print("Std:", X_scaled.std())
