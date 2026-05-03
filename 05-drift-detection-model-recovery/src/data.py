from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
