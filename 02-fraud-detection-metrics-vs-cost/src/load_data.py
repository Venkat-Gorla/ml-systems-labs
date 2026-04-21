from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"amount", "is_international", "is_night", "label"}


def load_fraud_data(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)

    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = REQUIRED_COLUMNS - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    X = df[["amount", "is_international", "is_night"]]
    y = df["label"]

    return X, y


def test_load_fraud_data():
    X, y = load_fraud_data("data/fraud.csv")

    assert not X.empty
    assert not y.empty

    assert list(X.columns) == ["amount", "is_international", "is_night"]
    assert len(X) == len(y)

    print("All load_fraud_data tests passed!")

    df = pd.read_csv("data/fraud.csv")
    print("\nSample data:")
    print(df.head())
    print(f"\n{df['label'].value_counts()}")


if __name__ == "__main__":
    test_load_fraud_data()
