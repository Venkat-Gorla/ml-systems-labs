from typing import Tuple, Dict, List, Set, Callable
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from data_loader import load_data
from evaluation import train_test_split, precision_at_k
from data_ops import add_negative_samples
from heuristic_model import build_user_item_map
from ml_utils import compute_item_popularity_feature, score_candidates
from debug_utils import debug_user_ml


def prepare_training_data(
    train_df: pd.DataFrame,
    use_item_popularity: bool,
) -> Tuple:
    """
    Prepare training data for naive ML model.

    Input:
        train_df with columns: user_id, item_id, interaction

    Output:
        X_encoded: feature matrix
        y: target vector
        encoder: fitted encoder (reuse for inference)
        item_popularity: Dict[item_id -> count]
    """

    X_raw = train_df[["user_id", "item_id"]]
    y = train_df["interaction"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(X_raw)

    if not use_item_popularity:
        return X_encoded, y, encoder, None

    popularity_values, item_popularity = compute_item_popularity_feature(
        train_df
    )
    X_encoded = np.hstack([X_encoded, popularity_values])

    return X_encoded, y, encoder, item_popularity


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
    item_popularity: Dict[int, int],
) -> Dict[int, List[int]]:
    """
    Generate top-K recommendations using ML model.

    Returns:
        {user_id: [item_id, ...]}
    """

    recommendations = {}

    for user, seen_items in user_items.items():
        scored_items = score_candidates(
            user,
            seen_items,
            all_items,
            model,
            encoder,
            item_popularity,
        )

        top_k = [int(item) for item, _ in scored_items[:k]]
        recommendations[user] = top_k

    return recommendations


def run_pipeline(
    data_path: str,
    negative_sampler: Callable[[pd.DataFrame], pd.DataFrame],
    k: int = 5,
    use_item_popularity: bool = False,
) -> float:
    df = train_test_split(load_data(data_path))
    train_df = df[~df["is_test"]]
    train_df = negative_sampler(train_df)
    test_df = df[df["is_test"]]

    train_user_items = build_user_item_map(train_df)
    test_user_items = build_user_item_map(test_df)

    X, y, encoder, item_popularity = prepare_training_data(
        train_df,
        use_item_popularity
    )
    model = train_model(X, y)

    all_items = sorted(df["item_id"].unique())

    # Recommend
    recommendations = recommend_top_k_ml(
        model,
        encoder,
        train_user_items,
        all_items,
        k,
        item_popularity,
    )

    debug_user_id = 1
    debug_user_ml(
        debug_user_id,
        train_user_items,
        test_user_items,
        model,
        encoder,
        item_popularity,
        all_items,
        k,
    )

    # Evaluate
    return precision_at_k(recommendations, test_user_items, k)


def main():
    k = 5
    score = run_pipeline(
        data_path="data/interactions.csv",
        negative_sampler=add_negative_samples,
        k=k,
    )
    print(f"ML Precision@{k}: {score:.4f}")


if __name__ == "__main__":
    main()
