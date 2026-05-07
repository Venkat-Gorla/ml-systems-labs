from __future__ import annotations

import os
import csv
from datetime import datetime, timezone

FILE_PATH = "docs/experiments.csv"


def log_experiment(
    experiment_type: str,
    noise: float,
    mae: float,
    rmse: float,
) -> None:
    """Persist regression experiment metrics."""

    os.makedirs("docs", exist_ok=True)

    file_exists = os.path.isfile(FILE_PATH)

    with open(FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "experiment_type",
                    "noise",
                    "mae",
                    "rmse",
                ]
            )

        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                experiment_type,
                noise,
                round(mae, 4),
                round(rmse, 4),
            ]
        )
