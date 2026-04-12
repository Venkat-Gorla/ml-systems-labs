import json
from pathlib import Path


def load_results():
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "results" / "metrics.json"

    with open(path, "r") as f:
        return json.load(f)


def print_comparison(results):
    print("\n=== Experiment Comparison ===")

    for r in results:
        m = r["metrics"]
        print(
            f"{r['model']} | {r['version']} | "
            f"precision={m['precision']} | recall={m['recall']}"
        )


def main():
    results = load_results()
    print_comparison(results)


if __name__ == "__main__":
    main()
