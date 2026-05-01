import random
from collections import Counter
from pathlib import Path
import pandas as pd

NUM_USERS = 80
NUM_ITEMS = 40
POS_PER_USER = 6
SEED = 42

OUTPUT_PATH = Path("data/interactions_v2.csv")


def assign_user_group(user_id: int) -> str:
    return "A" if user_id <= 40 else "B"


def assign_item_group(item_id: int) -> str:
    return "A" if item_id <= 20 else "B"


def generate_interactions() -> pd.DataFrame:
    random.seed(SEED)

    rows = []

    items_group_a = list(range(1, 21))
    items_group_b = list(range(21, 41))

    for user_id in range(1, NUM_USERS + 1):
        user_group = assign_user_group(user_id)

        if user_group == "A":
            preferred_items = items_group_a
            non_preferred_items = items_group_b
        else:
            preferred_items = items_group_b
            non_preferred_items = items_group_a

        positives_pref = random.sample(preferred_items, 5)
        positives_noise = random.sample(non_preferred_items, 1)

        for item_id in positives_pref + positives_noise:
            rows.append((user_id, item_id, 1))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "interaction"])
    return df


def debug_distribution(df: pd.DataFrame) -> None:
    counter = Counter()

    for _, row in df.iterrows():
        user_group = assign_user_group(row["user_id"])
        item_group = assign_item_group(row["item_id"])
        counter[(user_group, item_group)] += 1

    print("\nDistribution (user_group → item_group):")
    for key, count in sorted(counter.items()):
        print(f"{key}: {count}")


def main():
    df = generate_interactions()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    debug_distribution(df)


if __name__ == "__main__":
    main()
