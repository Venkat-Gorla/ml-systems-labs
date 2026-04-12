import json
from pathlib import Path
from datetime import datetime, timezone


def save_metrics(metrics: dict, model_name: str, version: str) -> None:
    base_dir = Path(__file__).resolve().parent.parent
    results_path = base_dir / "results" / "metrics.json"

    results_path.parent.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "version": version,
        "metrics": metrics,
    }

    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
