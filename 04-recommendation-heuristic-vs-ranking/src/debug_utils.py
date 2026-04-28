from typing import Dict, List, Set
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


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


# vegorla: fix duplicate code
def debug_user_ml(
    user_id: int,
    train_user_items: Dict[int, Set[int]],
    test_user_items: Dict[int, Set[int]],
    model: LogisticRegression,
    encoder: OneHotEncoder,
    all_items: List[int],
    k: int,
    top_n: int = 10,
):
    """
    Debug ML recommendation flow for a single user.

    Prints:
        - seen items
        - test item
        - top-N items by score
        - top-K recommendations
        - hits + precision
    """

    print("=" * 50)
    print(f"DEBUG ML USER: {user_id}")

    seen_items = train_user_items.get(user_id, set())
    test_items = test_user_items.get(user_id, set())

    print(f"Train items (seen): {sorted(seen_items)}")
    print(f"Test item: {list(test_items)}")

    # candidates
    candidates = [item for item in all_items if item not in seen_items]

    # build input
    X_input = pd.DataFrame({
        "user_id": [user_id] * len(candidates),
        "item_id": candidates,
    })

    # encode + score
    X_encoded = encoder.transform(X_input)
    probs = model.predict_proba(X_encoded)[:, 1]

    # pair + sort
    scored_items = sorted(
        zip(candidates, probs),
        key=lambda x: x[1],
        reverse=True,
    )

    # top-N (for inspection)
    print(f"\nTop-{top_n} items by score:")
    for item, score in scored_items[:top_n]:
        print(f"  item={int(item)}  score={score:.4f}")

    # top-K recommendations
    top_k_items = [int(item) for item, _ in scored_items[:k]]
    print(f"\nTop-{k} recommendations: {top_k_items}")

    # hits
    hits = [item for item in top_k_items if item in test_items]
    precision = len(hits) / k

    print(f"Hits: {hits} (count={len(hits)}, precision={precision:.2f})")
    print("=" * 50)
