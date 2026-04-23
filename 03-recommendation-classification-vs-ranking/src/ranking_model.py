from load_data import load_recommendation_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def split_data(X, y):
    return train_test_split(X, y, test_size=0.5, random_state=42)


def train_model(X_train, y_train):
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def predict_scores(model, X_test):
    return model.predict_proba(X_test)[:, 1]


def build_ranking_df(X_test, y_test, scores):
    df = X_test.copy()
    df["label"] = y_test.values
    df["score"] = scores
    return df


def filter_users(df):
    grouped = df.groupby("user_id")["item_id"].count()
    valid_users = grouped[grouped >= 2].index
    return df[df["user_id"].isin(valid_users)]


def precision_at_k(df, k=2):
    precisions = []

    for user_id, group in df.groupby("user_id"):
        group_sorted = group.sort_values("score", ascending=False)
        top_k = group_sorted.head(k)

        relevant = top_k["label"].sum()
        precision = relevant / k

        precisions.append(precision)

    return sum(precisions) / len(precisions)


def debug_print(df, k=2):
    for user_id, group in df.groupby("user_id"):
        print(f"\nUser {user_id}")

        group_sorted = group.sort_values("score", ascending=False)
        print(group_sorted[["item_id", "score", "label"]].head(k))


def test_ranking_model():
    X, y = load_recommendation_data("data/recommendation.csv")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    scores = predict_scores(model, X_test)

    df = build_ranking_df(X_test, y_test, scores)
    df = filter_users(df)
    debug_print(df, k=2)

    print("\nRanking Model Metrics:")
    p_at_2 = precision_at_k(df, k=2)
    print(f"Precision@2: {p_at_2:.3f}")

    assert 0 <= p_at_2 <= 1
    print("Ranking model test passed!")


if __name__ == "__main__":
    test_ranking_model()
