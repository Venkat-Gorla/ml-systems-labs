from ml_model import run_pipeline
from negative_sampling_v2 import add_negative_samples_v2


# vegorla: use command line parameter for item popularity
def main():
    score = run_pipeline(
        data_path="data/interactions_v2.csv",
        negative_sampler=add_negative_samples_v2,
        use_item_popularity=True,
    )
    print(f"ML Precision@5: {score:.4f}")


if __name__ == "__main__":
    main()
