import pandas as pd
import numpy as np
from pathlib import Path


def attach_and_persist_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: str = "data/processed/anomaly_results.csv"
) -> pd.DataFrame:
    """
    Attach anomaly labels and scores to dataframe and persist results.
    """
    df_out = df.copy()

    df_out["AnomalyLabel"] = labels
    df_out["AnomalyScore"] = scores

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_path, index=False)

    print(f"\nSaved results → {output_path}")

    return df_out
