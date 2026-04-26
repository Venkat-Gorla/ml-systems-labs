from load_and_inspect_data import load_and_validate_data
from features import scale_features


def main():
    path = "data/synthetic_manual.csv"
    df = load_and_validate_data(path)
    X_scaled = scale_features(df)

    print("\nPipeline step complete: features ready for anomaly detection")


if __name__ == "__main__":
    main()
