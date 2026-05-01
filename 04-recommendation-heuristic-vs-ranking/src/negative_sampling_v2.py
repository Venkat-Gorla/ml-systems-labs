import numpy as np
import pandas as pd
from typing import List


def assign_user_group(user_id: int) -> str:
    return "A" if user_id <= 40 else "B"


def get_group_items(group: str) -> List[int]:
    if group == "A":
        return list(range(1, 21))
    return list(range(21, 41))


def add_negative_samples_v2(
    train_df: pd.DataFrame,
    num_neg: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    user_items = train_df.groupby("user_id")["item_id"].apply(set)

    negative_rows = []

    for user, seen_items in user_items.items():
        user_group = assign_user_group(user)

        preferred_items = set(get_group_items(user_group))
        non_preferred_items = set(range(1, 41)) - preferred_items

        candidates = list(non_preferred_items - seen_items)

        if not candidates:
            continue

        sampled_items = rng.choice(
            candidates,
            size=min(num_neg, len(candidates)),
            replace=False,
        )

        for item in sampled_items:
            negative_rows.append({
                "user_id": user,
                "item_id": int(item),
                "interaction": 0,
                "is_test": False,
            })

    neg_df = pd.DataFrame(negative_rows)

    return pd.concat([train_df, neg_df], ignore_index=True)
