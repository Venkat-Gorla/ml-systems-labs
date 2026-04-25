from load_and_inspect_data import load_and_validate
from features import select_features, scale_features
from model import train_kmeans, inspect_clusters, interpret_clusters
from persist import save_clustered_data


def main():
    df = load_and_validate()

    X = select_features(df)
    X_scaled = scale_features(X)

    model = train_kmeans(X_scaled, k=5)
    inspect_clusters(model)

    print("\nPipeline step complete: clustering done")

    df_with_clusters = interpret_clusters(df, model.labels_)

    save_clustered_data(
        df_with_clusters,
        path="data/processed/customers_clustered.csv"
    )


if __name__ == "__main__":
    main()
