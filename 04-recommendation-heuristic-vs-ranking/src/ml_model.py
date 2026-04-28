from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def prepare_training_data(
    train_df: pd.DataFrame,
) -> Tuple:
    """
    Prepare training data for naive ML model.

    Input:
        train_df with columns: user_id, item_id, interaction

    Output:
        X_encoded: feature matrix
        y: target vector
        encoder: fitted encoder (reuse for inference)
    """

    X_raw = train_df[["user_id", "item_id"]]
    y = train_df["interaction"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_encoded = encoder.fit_transform(X_raw)

    return X_encoded, y, encoder


def train_model(
    X: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """
    Train Logistic Regression model.

    Input:
        X: encoded feature matrix
        y: interaction labels

    Output:
        trained LogisticRegression model
    """

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model
