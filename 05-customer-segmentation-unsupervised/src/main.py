import pandas as pd
from load_and_inspect_data import inspect_data
from features import select_features, scale_features


def main():
    df = pd.read_csv("data/Mall_Customers.csv")

    inspect_data(df)

    X = select_features(df)
    X_scaled = scale_features(X)

    print("\nPipeline step complete: features ready for clustering")


if __name__ == "__main__":
    main()
