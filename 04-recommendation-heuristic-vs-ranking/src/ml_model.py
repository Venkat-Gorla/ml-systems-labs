from typing import Tuple
from typing import Dict, List, Set
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from data_loader import load_data
from evaluation import train_test_split, precision_at_k
from heuristic_model import build_user_item_map
from debug_utils import debug_user


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


# vegorla: move to separate file and have sanity test
def add_negative_samples(
    train_df: pd.DataFrame,
    num_neg: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add negative samples (interaction=0) per user.

    Returns:
        DataFrame with both positive and negative samples
    """

    rng = np.random.default_rng(random_state)

    all_items: List[int] = train_df["item_id"].unique().tolist()
    user_items = train_df.groupby("user_id")["item_id"].apply(set)

    negative_rows = []

    for user, seen_items in user_items.items():
        unseen_items = list(set(all_items) - seen_items)

        if len(unseen_items) == 0:
            continue

        sampled_items = rng.choice(
            unseen_items,
            size=min(num_neg, len(unseen_items)),
            replace=False,
        )

        for item in sampled_items:
            negative_rows.append(
                {"user_id": user, "item_id": item, "interaction": 0}
            )

    neg_df = pd.DataFrame(negative_rows)

    return pd.concat([train_df, neg_df], ignore_index=True)


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
    debug_user(
        debug_user_id,
        train_user_items,
        test_user_items,
        pd.Series(dtype=float),  # placeholder (no popularity here)
        recommendations,
    )

    # Evaluate
    score = precision_at_k(recommendations, test_user_items, k)
    print(f"ML Precision@{k}: {score:.4f}")


if __name__ == "__main__":
    main()
