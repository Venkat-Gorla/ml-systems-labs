import argparse
from ml_model import run_pipeline
from negative_sampling_v2 import add_negative_samples_v2


def main():
    args = parse_args()

    score = run_pipeline(
        data_path="data/interactions_v2.csv",
        negative_sampler=add_negative_samples_v2,
        use_item_popularity=args.use_popularity,
    )
    print(f"ML Precision@5: {score:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-popularity",
        action="store_true",
        help="Enable item popularity feature",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
