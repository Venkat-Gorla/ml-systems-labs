"""
uv run python -m src.load_and_inspect_data
"""
import pandas as pd


REQUIRED_COLUMNS = ["Feature1", "Feature2"]


def load_and_validate_data(path: str) -> pd.DataFrame:
    """
    Load dataset and perform strict validation checks.
    """
    df = pd.read_csv(path)

    assert not df.empty, "Dataset is empty"

    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"

    assert df[REQUIRED_COLUMNS].isnull().sum().sum() == 0, "Null values found"

    # Type check (must be numeric)
    for col in REQUIRED_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"

    return df


def inspect_data(path: str) -> pd.DataFrame:
    """
    Load data and print key diagnostics for understanding.
    """
    df = load_and_validate_data(path)

    print("\n=== Data Shape ===")
    print(df.shape)

    print("\n=== Sample Data ===")
    print(df.head())

    print("\n=== Summary Stats ===")
    print(df.describe())

    print("\n=== Min / Max Check (important for anomalies) ===")
    print(df.agg(["min", "max"]))

    print("\nAll validations passed!")

    return df


def main():
    path = "data/synthetic_manual.csv"
    inspect_data(path)


if __name__ == "__main__":
    main()
