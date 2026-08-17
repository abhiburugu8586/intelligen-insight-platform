"""
Finds a good value for n_clusters using silhouette score (a standard,
defensible way to justify the choice in your report, rather than picking a
number arbitrarily), then generates a finished cluster analysis summary and
chart ready for the report/video.

Usage:
    python -m src.finalize_clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.clustering import ReviewClusterer


def find_best_k(texts, k_range=range(2, 11)):
    """
    Tries several values of k and scores each with silhouette score
    (higher is better, range -1 to 1). This gives a defensible, data-driven
    justification for the final n_clusters choice instead of guessing.
    """
    results = []
    for k in k_range:
        clusterer = ReviewClusterer(n_clusters=k)
        labels = clusterer.fit(texts)
        score = silhouette_score(clusterer._embeddings, labels)
        results.append({"k": k, "silhouette_score": score})
        print(f"k={k}: silhouette score = {score:.3f}")
    return pd.DataFrame(results)


def main():
    df = pd.read_csv("data/sample_reviews.csv")
    texts = df["text"].astype(str).tolist()

    print(f"Testing cluster counts on {len(texts)} reviews...\n")
    scores_df = find_best_k(texts)

    best_k = int(scores_df.loc[scores_df["silhouette_score"].idxmax(), "k"])
    print(f"\nBest k based on silhouette score: {best_k}")

    # Save the score comparison chart - useful evidence for the report
    plt.figure(figsize=(6, 4))
    plt.plot(scores_df["k"], scores_df["silhouette_score"], marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"chosen k={best_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Cluster count selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/cluster_k_selection.png")
    print("Saved chart to data/cluster_k_selection.png")

    # Now run final clustering with the chosen k and save a theme summary
    clusterer = ReviewClusterer(n_clusters=best_k)
    labels = clusterer.fit(texts)
    themes = clusterer.top_terms_per_cluster(texts, labels)

    df["cluster"] = labels
    df["cluster_theme"] = df["cluster"].map(lambda c: ", ".join(themes.get(c, [])))
    df.to_csv("data/clustered_reviews.csv", index=False)

    print(f"\nFinal clustering (k={best_k}):")
    for cluster_id, terms in themes.items():
        count = int((df["cluster"] == cluster_id).sum())
        print(f"  Cluster {cluster_id} ({count} reviews): {', '.join(terms)}")

    print("\nSaved data/clustered_reviews.csv with cluster assignments.")
    print(f"\nUpdate src/clustering.py's default n_clusters to {best_k} for the final app.")


if __name__ == "__main__":
    main()