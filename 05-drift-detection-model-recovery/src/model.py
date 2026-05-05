from sklearn.linear_model import LogisticRegression
import numpy as np


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model):
    coefficients = model.coef_[0]  # assuming binary classification
    importance = np.abs(coefficients)

    ranked_indices = np.argsort(importance)[::-1]

    return {
        "ranked_indices": ranked_indices,
        "importance": importance,
    }


def test_get_feature_importance():
    from data import load_data, split_data

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    result = get_feature_importance(model)
    top_features = result["ranked_indices"][:3]

    print("\nTop important features:")
    for idx in top_features:
        print(f"Feature {idx}: {result['importance'][idx]:.4f}")


if __name__ == "__main__":
    test_get_feature_importance()
