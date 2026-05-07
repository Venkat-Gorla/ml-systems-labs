# Anomaly Detection — Pipeline Output

## Command

```bash
uv run src/main.py
```

## Output

```
=== Scaling Check ===
Mean (approx 0): [6.66133815e-17 2.96059473e-17]
Std (approx 1): [1. 1.]

=== Prediction Sample (first 10) ===
[1 1 1 1 1 1 1 1 1 1]

=== Prediction Counts ===
Anomaly: 6
Normal: 54

=== Last 10 Rows (Validation) ===
[ 1  1 -1 -1 -1  1  1 -1 -1 -1]

=== Score Sample (first 10) ===
[0.14757642 0.17059464 0.09797252 0.14426138 0.09852164 0.04973238
 0.15370034 0.15067724 0.15023464 0.15361068]

=== Top 10 Most Anomalous Points (by score) ===
[59 54 52 58 53 57 55 56 51 50]

Corresponding Scores:
[-0.12526556 -0.0543603  -0.04920848 -0.04432052 -0.00523038 -0.00338973
  0.00037664  0.00179627  0.0033807   0.00451286]

Pipeline step complete: anomaly detection done
```

## Interpretation

- Model detects anomalies using Isolation Forest
- Contamination set to 0.1 → ~10% anomalies expected
- 6 anomalies identified out of 60 data points

### Key Observations

- Last rows contain most anomalies (by design of synthetic dataset)
- Score-based ranking confirms correct ordering:
  - Lower score → more anomalous
- Top anomalous indices align with expected injected anomalies

## Output Artifact

```
data/processed/anomaly_results.csv
```

### Contains

- Original features
- `AnomalyLabel` (1 = normal, -1 = anomaly)
- `AnomalyScore` (continuous score)
