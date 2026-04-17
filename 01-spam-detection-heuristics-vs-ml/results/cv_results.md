# Cross Validation Results

## Model

- TF-IDF + Logistic Regression
- 5-fold Stratified Cross Validation

## Results

| Metric    | Single Split | CV Mean ± Std   |
| --------- | ------------ | --------------- |
| Accuracy  | 0.9704       | 0.9659 ± 0.0034 |
| Precision | 1.0000       | 0.9859 ± 0.0090 |
| Recall    | 0.7785       | 0.7563 ± 0.0205 |

## Observations

- Single split results are close to cross-validation mean → model is stable
- Low standard deviation across folds → consistent performance
- Precision is very high (near perfect) → almost no false positives
- Recall is lower → model is missing some spam messages

## Conclusion

- Model is reliable but conservative
- Main improvement area: increase recall without significantly reducing precision

## Next Steps

- Threshold tuning to improve recall
- Feature engineering (e.g., n-grams)
