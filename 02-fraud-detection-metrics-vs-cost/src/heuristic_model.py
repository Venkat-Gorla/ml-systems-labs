from load_data import load_fraud_data
from data_ops import split_data, evaluate


def baseline_predict(X_test):
    return [0] * len(X_test)


def test_baseline_model():
    X, y = load_fraud_data("data/fraud.csv")
    _, X_test, _, y_test = split_data(X, y)

    y_pred = baseline_predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print("\nBaseline Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    assert metrics["recall"] == 0.0  # baseline misses all fraud
    assert metrics["precision"] == 0.0  # no positive predictions
    assert all(p == 0 for p in y_pred)

    print("Baseline test passed!")


if __name__ == "__main__":
    test_baseline_model()
