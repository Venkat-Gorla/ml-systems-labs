from typing import Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def run_cross_validation(X, y, folds: int = 5) -> Dict[str, float]:
    """
    Run cross-validation using Pipeline (TF-IDF + Logistic Regression).
    """

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    results = {}

    print(f"\n=== Cross Validation ({folds}-fold) ===")

    for metric in scoring.keys():
        values = scores[f"test_{metric}"]
        mean = np.mean(values)
        std = np.std(values)

        print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")

        results[f"{metric}_mean"] = float(mean)
        results[f"{metric}_std"] = float(std)

    return results
