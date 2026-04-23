from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"user_id", "item_id", "interaction"}


def load_recommendation_data(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)

    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = REQUIRED_COLUMNS - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    # ranking requires both positive and negative per user
    grouped = df.groupby("user_id")["interaction"].nunique()
    invalid_users = grouped[grouped < 2]

    if not invalid_users.empty:
        raise ValueError(
            f"Users with only one class: {list(invalid_users.index)}")

    X = df[["user_id", "item_id"]]
    y = df["interaction"]

    return X, y


def test_load_recommendation_data():
    X, y = load_recommendation_data("data/recommendation.csv")

    assert not X.empty
    assert not y.empty

    assert list(X.columns) == ["user_id", "item_id"]
    assert len(X) == len(y)

    print("All load_recommendation_data tests passed!")

    df = pd.read_csv("data/recommendation.csv")
    print("\nSample data:")
    print(df.head())
    print("\nPer-user interaction counts:")
    print(df.groupby("user_id")["interaction"].value_counts())


if __name__ == "__main__":
    test_load_recommendation_data()
