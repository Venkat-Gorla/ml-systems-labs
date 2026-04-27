from typing import Dict, List
import pandas as pd
from data_loader import load_data
from evaluation import train_test_split, precision_at_k


def build_user_item_map(df: pd.DataFrame) -> Dict[int, set]:
    """Return mapping: {user_id: set(item_id, ...)}"""
    user_items = (
        df[df["interaction"] == 1]
        .groupby("user_id")["item_id"]
        .apply(set)
    )
    return user_items.to_dict()


def compute_item_popularity(df: pd.DataFrame) -> pd.Series:
    return df.groupby("item_id")["interaction"].sum().sort_values(ascending=False)


def recommend_top_k(
    user_items: Dict[int, set],
    item_popularity: pd.Series,
    k: int,
) -> Dict[int, List[int]]:
    all_items_ranked = item_popularity.index.tolist()
    recommendations = {}

    for user, seen_items in user_items.items():
        recs = []
        for item in all_items_ranked:
            if item not in seen_items:
                recs.append(item)
            if len(recs) == k:
                break
        recommendations[user] = recs

    return recommendations


def main():
    path = "data/interactions.csv"
    k = 5

    interactions_df = load_data(path)
    split_df = train_test_split(interactions_df)

    train_df = split_df[split_df["is_test"] == False]
    test_df = split_df[split_df["is_test"] == True]

    train_user_items = build_user_item_map(train_df)
    item_popularity = compute_item_popularity(train_df)

    recommendations = recommend_top_k(train_user_items, item_popularity, k)

    test_user_items = build_user_item_map(test_df)
    score = precision_at_k(recommendations, test_user_items, k)

    print(f"Precision@{k}: {score:.4f}")


if __name__ == "__main__":
    main()
