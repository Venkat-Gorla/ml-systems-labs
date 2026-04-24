import pandas as pd


REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]


# vegorla: package this logic into reusable function
def inspect_data(df: pd.DataFrame) -> None:
    print("\n=== Data Shape ===")
    print(df.shape)

    print("\n=== Sample Data ===")
    print(df.head())

    print("\n=== Column Check ===")
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"
    print("All required columns present")

    print("\n=== Null Check ===")
    null_counts = df.isnull().sum()
    print(null_counts)
    assert null_counts.sum() == 0, "Null values found"

    print("\n=== Gender Distribution ===")
    print(df["Gender"].value_counts())

    print("\n=== Spending Score Validation ===")
    assert df["Spending Score (1-100)"].between(1, 100).all()
    print("Spending score within expected range")

    print("\n=== Numeric Columns Check ===")
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    assert df[numeric_cols].select_dtypes(
        include="number").shape[1] == len(numeric_cols)
    print("All required numeric columns are valid")

    print("\n=== Duplicate Check ===")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    print("\n=== Summary Stats ===")
    print(df.describe())

    print("\nAll validations for inspect_data passed!")


def main():
    df = pd.read_csv("data/Mall_Customers.csv")
    inspect_data(df)


if __name__ == "__main__":
    main()
