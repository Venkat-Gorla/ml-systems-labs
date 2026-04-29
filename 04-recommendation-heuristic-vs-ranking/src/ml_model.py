from typing import Tuple
from typing import Dict, List, Set
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from data_loader import load_data
from evaluation import train_test_split, precision_at_k
from data_ops import add_negative_samples
from heuristic_model import build_user_item_map
from ml_utils import score_candidates
from debug_utils import debug_user_ml


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

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
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
        scored_items = score_candidates(
            user, seen_items, all_items, model, encoder
        )

        top_k = [int(item) for item, _ in scored_items[:k]]
        recommendations[user] = top_k

    return recommendations


def main():
    path = "data/interactions.csv"
    k = 5

    df = train_test_split(load_data(path))
    train_df = df[~df["is_test"]]
    train_df = add_negative_samples(train_df)
    test_df = df[df["is_test"]]

    train_user_items = build_user_item_map(train_df)
    test_user_items = build_user_item_map(test_df)

    X, y, encoder = prepare_training_data(train_df)
    model = train_model(X, y)

    all_items = sorted(df["item_id"].unique())

    # Recommend
    recommendations = recommend_top_k_ml(
        model,
        encoder,
        train_user_items,
        all_items,
        k,
    )

    debug_user_id = 1
    debug_user_ml(
        debug_user_id,
        train_user_items,
        test_user_items,
        model,
        encoder,
        all_items,
        k,
    )

    # Evaluate
    score = precision_at_k(recommendations, test_user_items, k)
    print(f"ML Precision@{k}: {score:.4f}")


if __name__ == "__main__":
    main()
