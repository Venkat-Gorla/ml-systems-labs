import pandas as pd
from pathlib import Path


def save_clustered_data(df: pd.DataFrame, path: str) -> None:
    """
    Save dataframe with cluster labels for downstream tasks (e.g., PCA).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nSaved clustered data → {output_path}")
