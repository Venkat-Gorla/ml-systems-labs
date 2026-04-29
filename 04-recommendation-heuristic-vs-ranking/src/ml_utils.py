from typing import List, Set, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def score_candidates(
    user_id: int,
    seen_items: Set[int],
    all_items: List[int],
    model: LogisticRegression,
    encoder: OneHotEncoder,
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

    X_encoded = encoder.transform(X_input)
    probs = model.predict_proba(X_encoded)[:, 1]

    # rank items by probability
    scored_items = sorted(
        zip(candidates, probs),
        key=lambda x: x[1],
        reverse=True,
    )

    return scored_items
