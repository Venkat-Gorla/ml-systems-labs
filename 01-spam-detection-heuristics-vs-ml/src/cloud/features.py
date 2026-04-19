import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(df: pd.DataFrame):
    X = df["message"]
    y = df["label"].str.lower().map({"ham": 0, "spam": 1})

    if y.isnull().any():
        raise ValueError("Unknown labels found during mapping")

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # keep same spam/ham ratio in train & test
    )


def fit_vectorizer(X_train):
    """
    Fit TF-IDF vectorizer on training data and return vectorizer + transformed data.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_vec


def transform(vectorizer, X):
    """
    Transform input text using a pre-fitted vectorizer.
    """
    return vectorizer.transform(X)


def vectorize(X_train, X_test):
    vectorizer, X_train_vec = fit_vectorizer(X_train)
    X_test_vec = transform(vectorizer, X_test)

    return X_train_vec, X_test_vec
