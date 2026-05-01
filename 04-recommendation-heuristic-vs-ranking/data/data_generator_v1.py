import random
from typing import List, Tuple
import pandas as pd


def generate_user_groups(num_users: int, num_groups: int) -> dict:
    users = list(range(1, num_users + 1))
    random.shuffle(users)

    group_size = num_users // num_groups
    user_groups = {}

    for i in range(num_groups):
        group_users = users[i * group_size: (i + 1) * group_size]
        for user in group_users:
            user_groups[user] = i

    return user_groups


def generate_group_item_preferences(
    num_items: int, num_groups: int
) -> dict:
    items = list(range(1, num_items + 1))
    group_size = num_items // num_groups

    group_items = {}

    for i in range(num_groups):
        group_items[i] = items[i * group_size: (i + 1) * group_size]

    return group_items


def generate_interactions(
    num_users: int,
    num_items: int,
    num_groups: int,
    interactions_per_user: int,
    noise_ratio: float = 0.2,
) -> pd.DataFrame:
    user_groups = generate_user_groups(num_users, num_groups)
    group_items = generate_group_item_preferences(num_items, num_groups)

    data: List[Tuple[int, int, int]] = []

    for user in range(1, num_users + 1):
        group = user_groups[user]
        preferred_items = group_items[group]

        num_noise = int(interactions_per_user * noise_ratio)
        num_pref = interactions_per_user - num_noise

        # Preferred interactions
        pref_items = random.sample(
            preferred_items, min(len(preferred_items), num_pref)
        )

        # Noise interactions (from other groups)
        non_pref_items = list(
            set(range(1, num_items + 1)) - set(preferred_items))
        noise_items = random.sample(
            non_pref_items, min(len(non_pref_items), num_noise)
        )

        interacted_items = pref_items + noise_items

        for item in interacted_items:
            data.append((user, item, 1))

    df = pd.DataFrame(data, columns=["user_id", "item_id", "interaction"])
    return df


def main():
    df = generate_interactions(
        num_users=80,
        num_items=40,
        num_groups=4,
        interactions_per_user=6,
        noise_ratio=0.2,
    )

    print("Sample data:")
    print(df.head())

    print("\nDataset stats:")
    print(f"Total rows: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique items: {df['item_id'].nunique()}")

    df.to_csv("data/interactions.csv", index=False)
    print("\nSaved to data/interactions.csv")


if __name__ == "__main__":
    main()
