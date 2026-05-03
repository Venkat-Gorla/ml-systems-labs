from typing import List, Set, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def score_candidates(
    user_id: int,
    seen_items: Set[int],
    all_items: List[int],
    model: LogisticRegression,
    encoder: OneHotEncoder,
    item_popularity: Dict[int, int],
) -> List[Tuple[int, float]]:
    """
    Returns:
        Sorted list of (item_id, score) in descending order
    """

    # candidates = all unseen items
    candidates = [item for item in all_items if item not in seen_items]

    X_input = pd.DataFrame({
        "user_id": [user_id] * len(candidates),
        "item_id": candidates,
    })

    X_cat = encoder.transform(X_input)

    if item_popularity is not None:
        pop_values = _get_popularity_values(X_input, item_popularity)
        X_encoded = np.hstack([X_cat, pop_values])
    else:
        X_encoded = X_cat

    expected_features = model.n_features_in_
    actual_features = X_encoded.shape[1]

    assert expected_features == actual_features, (
        f"Feature mismatch: expected={expected_features}, got={actual_features}"
    )

    probs = model.predict_proba(X_encoded)[:, 1]

    # rank items by probability
    scored_items = sorted(
        zip(candidates, probs),
        key=lambda x: x[1],
        reverse=True,
    )

    return scored_items


def compute_item_popularity_feature(
    train_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Compute item popularity feature.

    Returns:
        popularity_values: (n, 1) numpy array aligned with train_df rows
        item_popularity: Dict[item_id -> count]
    """

    item_popularity = (
        train_df[train_df["interaction"] == 1]
        .groupby("item_id")
        .size()
        .to_dict()
    )

    popularity_values = _get_popularity_values(train_df, item_popularity)

    return popularity_values, item_popularity


def _get_popularity_values(X_input, item_popularity):
    pop_values = (
        X_input["item_id"]
        .map(item_popularity)
        .fillna(0)
        .values.reshape(-1, 1)
    )
    return pop_values


# --- driver / debug ---
def _debug_item_popularity():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "item_id": [10, 10, 20, 20, 30],
        "interaction": [1, 1, 1, 0, 0],
    })

    popularity_values, item_popularity = compute_item_popularity_feature(df)

    print("Item popularity:", item_popularity)
    print("Pop values:\n", popularity_values)

    assert item_popularity[10] == 2
    assert item_popularity[20] == 1
    assert item_popularity.get(30, 0) == 0

    # row-wise check
    expected = [2, 2, 1, 1, 0]
    assert popularity_values.flatten().tolist() == expected

    print("All tests for compute_item_popularity_feature() passed ✅")


if __name__ == "__main__":
    _debug_item_popularity()
