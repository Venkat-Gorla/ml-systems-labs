from typing import Tuple
from typing import Dict, List, Set
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def prepare_training_data(
    train_df: pd.DataFrame,
) -> Tuple:
    """
    Prepare training data for naive ML model.

    Input:
        train_df with columns: user_id, item_id, interaction

    Output:
        X_encoded: feature matrix
        y: target vector
        encoder: fitted encoder (reuse for inference)
    """

    X_raw = train_df[["user_id", "item_id"]]
    y = train_df["interaction"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_encoded = encoder.fit_transform(X_raw)

    return X_encoded, y, encoder


def train_model(
    X: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """
    Train Logistic Regression model.

    Input:
        X: encoded feature matrix
        y: interaction labels

    Output:
        trained LogisticRegression model
    """

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model


def recommend_top_k_ml(
    model: LogisticRegression,
    encoder: OneHotEncoder,
    user_items: Dict[int, Set[int]],
    all_items: List[int],
    k: int,
) -> Dict[int, List[int]]:
    """
    Generate top-K recommendations using ML model.

    Returns:
        {user_id: [item_id, ...]}
    """

    recommendations = {}

    for user, seen_items in user_items.items():
        # candidates = all unseen items
        candidates = [item for item in all_items if item not in seen_items]

        X_input = pd.DataFrame({
            "user_id": [user] * len(candidates),
            "item_id": candidates,
        })

        X_encoded = encoder.transform(X_input)

        probs = model.predict_proba(X_encoded)[:, 1]

        # rank items by probability
        ranked_items = [
            item for _, item in sorted(
                zip(probs, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        recommendations[user] = ranked_items[:k]

    return recommendations
