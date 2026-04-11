import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin-1")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("\n=== Sample Data ===")
    print(df.head())

    print("\n=== Columns ===")
    print(df.columns)

    print("\n=== Class Distribution ===")
    print(df['label'].value_counts())

    print("\n=== Example Messages ===")
    print(df['message'].sample(5).values)


def main():
    file_path = "data/spam.csv"
    df = load_data(file_path)
    inspect_data(df)


if __name__ == "__main__":
    main()
