from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    model = LogisticRegression()

    # why: small dataset + possible imbalance
    model.fit(X_train, y_train)

    return model


def predict(model, X, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def test_ml_model():
    from load_data import load_fraud_data
    from data_ops import split_data, evaluate

    X, y = load_fraud_data("data/fraud.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)

    metrics = evaluate(y_test, y_pred)

    print("\nML Model Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # sanity checks (not strict expectations)
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})

    print("ML model test passed!")


if __name__ == "__main__":
    test_ml_model()
