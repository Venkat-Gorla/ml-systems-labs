import pandas as pd
from load_and_inspect_data import load_and_validate
from features import select_features, scale_features


def main():
    df = load_and_validate()

    X = select_features(df)
    X_scaled = scale_features(X)

    print("\nPipeline step complete: features ready for clustering")


if __name__ == "__main__":
    main()
