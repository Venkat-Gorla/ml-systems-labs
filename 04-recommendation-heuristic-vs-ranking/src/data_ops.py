import pandas as pd
import numpy as np
from typing import List

from data_loader import load_data
from evaluation import train_test_split


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
            negative_rows.append({
                "user_id": user,
                "item_id": item,
                "interaction": 0,
                "is_test": False,
            })

    neg_df = pd.DataFrame(negative_rows)

    return pd.concat([train_df, neg_df], ignore_index=True)


def test_add_negative_samples():
    path = "data/interactions.csv"
    num_neg = 3

    # Load + prepare input
    df = train_test_split(load_data(path))
    train_df = df[~df["is_test"]]
    train_df_before = train_df.copy()

    # Preconditions
    assert train_df["interaction"].eq(
        1).all(), "Train data must be all positives"
    assert len(train_df) == len(train_df.drop_duplicates()
                                ), "Duplicates found in train data"

    expected_rows = train_df["user_id"].nunique() * 5
    assert len(train_df) == expected_rows, "Unexpected train row count"

    # Run function
    result_df = add_negative_samples(train_df, num_neg=num_neg)
    assert train_df.equals(train_df_before), "Input train_df was modified"

    # Postconditions
    # 1. Original data preserved
    merged = result_df.merge(
        train_df_before,
        on=["user_id", "item_id", "interaction"],
        how="inner",
    )
    assert len(merged) == len(
        train_df_before), "Original rows lost or modified"

    # 2. Split positives / negatives
    pos_df = result_df[result_df["interaction"] == 1]
    neg_df = result_df[result_df["interaction"] == 0]

    # 3. Positive count unchanged
    assert len(pos_df) == len(train_df_before), "Positive rows mismatch"

    # 4. Negative count exact
    num_users = train_df["user_id"].nunique()
    assert len(neg_df) == num_users * \
        num_neg, "Incorrect number of negative samples"

    # 5. Negatives unique
    assert len(neg_df) == len(neg_df.drop_duplicates()
                              ), "Duplicate negatives found"

    # 6. Per-user checks
    user_pos = pos_df.groupby("user_id")["item_id"].apply(set)
    user_neg = neg_df.groupby("user_id")["item_id"].apply(set)

    for user in user_pos.index:
        pos_items = user_pos[user]
        neg_items = user_neg[user]

        # exact count
        assert len(
            neg_items) == num_neg, f"User {user} does not have {num_neg} negatives"

        # no overlap
        assert pos_items.isdisjoint(
            neg_items), f"Overlap found for user {user}"

    # Debug print (user 1)
    user_id = 1
    print("\nDEBUG: Negative samples for user 1")
    print(neg_df[neg_df["user_id"] == user_id])

    print("\nAll tests for add_negative_samples() passed ✔")


if __name__ == "__main__":
    test_add_negative_samples()
