from typing import Dict, List, Set
import pandas as pd


def debug_user(
    user_id: int,
    train_user_items: Dict[int, Set[int]],
    test_user_items: Dict[int, Set[int]],
    item_popularity: pd.Series,
    recommendations: Dict[int, List[int]],
):
    print("\n" + "=" * 50)
    print(f"DEBUG USER: {user_id}")

    seen = train_user_items.get(user_id, set())
    test = test_user_items.get(user_id, set())

    print(f"Train items (seen): {sorted(seen)}")
    print(f"Test item: {sorted(test)}")

    ranked_items = item_popularity.index.tolist()
    candidates = [item for item in ranked_items if item not in seen]

    print(f"Top 10 popular items: {ranked_items[:10]}")
    print(f"Candidate items (first 10): {candidates[:10]}")

    recs = recommendations[user_id]
    print(f"Top-K recommendations: {recs}")

    hits = [item for item in recs if item in test]
    hit_count = len(hits)
    print(
        f"Hits: {hits} (count={hit_count}, precision={hit_count/len(recs):.2f})")

    print("=" * 50 + "\n")
