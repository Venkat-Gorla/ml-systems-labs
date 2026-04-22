from load_data import load_fraud_data
from data_ops import split_data, evaluate, print_metrics
from ml_model import train_model, predict


def run_threshold_experiment():
    X, y = load_fraud_data("data/fraud.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    thresholds = [0.1, 0.3, 0.5, 0.7]

    for t in thresholds:
        print(f"\nThreshold: {t}")

        y_pred = predict(model, X_test, threshold=t)
        metrics = evaluate(y_test, y_pred)

        print_metrics(metrics)
        print(f"predicted positives: {sum(y_pred)}")


if __name__ == "__main__":
    run_threshold_experiment()
