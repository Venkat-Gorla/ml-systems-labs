from typing import Dict, List
import pandas as pd


def train_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform leave-one-out split for recommendation evaluation.

    For each user:
    - Randomly select 1 interaction as test
    - Remaining interactions are used for training

    This simulates:
    "Given past interactions → predict next item"

    Steps:
    1. Shuffle dataset (to randomize selection)
    2. Pick first row per user → test
    3. Mark others → train

    Example:
    Input:
        user 1 → [A, B, C]

    After split:
        Train → [A, C]
        Test  → [B]

    Notes:
    - frac=1 means full shuffle
    - Ensures unbiased test selection per user
    """

    df = df.sample(frac=1, random_state=42)

    test_idx = df.groupby("user_id").head(1).index
    df["is_test"] = False
    df.loc[test_idx, "is_test"] = True

    return df


def precision_at_k(
    recommendations: Dict[int, List[int]],
    user_items: Dict[int, set],
    k: int,
) -> float:
    """
    Compute average Precision@K across users.

    For each user:
    - Look at top-K recommended items
    - Count how many are actually relevant (present in ground truth)
    - Precision@K = hits / K

    Final score = mean Precision@K over all users

    Example 1:
        recommendations = {1: [10, 20, 30]}
        user_items      = {1: {20}}

        hits = 1 (item 20)
        Precision@3 = 1 / 3 = 0.33

    Example 2:
        recommendations = {1: [10, 20, 30], 2: [5, 6, 7]}
        user_items      = {1: {20},          2: {8}}

        User 1 → hit (20) → 1/3
        User 2 → no hit   → 0/3

        Final = (1/3 + 0) / 2 = 0.1667

    Notes:
    - K is fixed for all users
    - If each user has 1 test item, max Precision@K = 1/K
    """

    precisions = []

    for user, recs in recommendations.items():
        relevant = user_items.get(user, set())
        hits = sum(1 for item in recs if item in relevant)
        precisions.append(hits / k)

    return sum(precisions) / len(precisions)
