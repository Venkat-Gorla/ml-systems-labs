from load_data import load_fraud_data
from data_ops import split_data
from ml_model import train_model, predict


def compute_confusion(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def run_cost_analysis():
    X, y = load_fraud_data("data/fraud.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    thresholds = [0.1, 0.3, 0.5, 0.7]

    cost_fp = 1
    cost_fn = 10

    for t in thresholds:
        print(f"\nThreshold: {t}")

        y_pred = predict(model, X_test, threshold=t)
        tp, tn, fp, fn = compute_confusion(y_test.values, y_pred)

        total_cost = fp * cost_fp + fn * cost_fn

        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Total Cost: {total_cost}")


if __name__ == "__main__":
    run_cost_analysis()
