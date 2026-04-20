from pathlib import Path

from load_and_inspect_data import load_data
from ml_pipeline import main


def run():
    data_path = Path("/opt/ml/input/data/train/spam.csv")
    df = load_data(data_path)

    main(
        save=False,
        confusion=False,
        cv=False,
        threshold=False,
        df=df,
    )


if __name__ == "__main__":
    run()
