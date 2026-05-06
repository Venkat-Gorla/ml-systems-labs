from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

FILE_PATH = "docs/experiments.csv"


def log_experiment(
    drift_type: str,
    drift_strength: float,
    features: list[int],
    precision: float,
    recall: float,
    f1: float,
    mean_psi: float = 0,
    max_psi: float = 0,
):
    os.makedirs("docs", exist_ok=True)  # ensure folder exists

    file_exists = os.path.isfile(FILE_PATH)

    with open(FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "drift_type",
                    "drift_strength",
                    "features",
                    "precision",
                    "recall",
                    "f1",
                    "mean_psi",
                    "max_psi",
                ]
            )

        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                drift_type,
                drift_strength,
                features,
                round(precision, 4),
                round(recall, 4),
                round(f1, 4),
                round(mean_psi, 4),
                round(max_psi, 4),
            ]
        )
