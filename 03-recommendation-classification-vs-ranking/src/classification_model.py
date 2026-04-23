from load_data import load_recommendation_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(X, y):
    return train_test_split(X, y, test_size=0.5, random_state=42)


def train_model(X_train, y_train):
    # model = LogisticRegression()
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def evaluate(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


def test_classification_model():
    X, y = load_recommendation_data("data/recommendation.csv")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)

    metrics = evaluate(y_test, y_pred)

    print("\nClassification Model Metrics:")
    print("Predicted positives:", sum(y_pred))
    print_metrics(metrics)

    assert len(y_pred) == len(y_test)
    print("Classification model test passed!")


if __name__ == "__main__":
    test_classification_model()
