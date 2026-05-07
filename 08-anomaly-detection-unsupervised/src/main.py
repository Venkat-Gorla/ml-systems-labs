from load_and_inspect_data import load_and_validate_data
from features import scale_features
from model import run_isolation_forest, print_anomaly_summary
from persist import attach_and_persist_results


def main():
    path = "data/synthetic_manual.csv"
    df = load_and_validate_data(path)
    X_scaled = scale_features(df)

    labels, scores = run_isolation_forest(X_scaled)
    print_anomaly_summary(labels, scores)

    attach_and_persist_results(df, labels, scores)


if __name__ == "__main__":
    main()
