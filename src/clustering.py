"""
Clustering: embeds reviews with a sentence-transformer model, then groups
them with K-Means to surface themes without needing predefined labels.
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class ReviewClusterer:
    def __init__(self, n_clusters: int = 7, embedding_model: str = "all-MiniLM-L6-v2"):
        # n_clusters=7 chosen via silhouette score analysis - see
        # src/finalize_clustering.py and data/cluster_k_selection.png
        self.n_clusters = n_clusters
        self.embedder = SentenceTransformer(embedding_model)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self._embeddings = None
        self._fitted = False

    def fit(self, texts: List[str]) -> np.ndarray:
        self._embeddings = self.embedder.encode(texts, show_progress_bar=False)
        self.kmeans.fit(self._embeddings)
        self._fitted = True
        return self.kmeans.labels_

    def predict(self, texts: List[str]) -> List[int]:
        if not self._fitted:
            raise RuntimeError("Call .fit() first")
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return list(self.kmeans.predict(embeddings))

    def top_terms_per_cluster(self, texts: List[str], labels: np.ndarray, top_n: int = 8) -> Dict[int, List[str]]:
        """Returns the most frequent non-stopword terms for each cluster."""
        from sklearn.feature_extraction.text import CountVectorizer

        themes: Dict[int, List[str]] = {}
        for cluster_id in sorted(set(labels)):
            cluster_texts = [t for t, l in zip(texts, labels) if l == cluster_id]
            if not cluster_texts:
                continue
            vec = CountVectorizer(stop_words="english", max_features=top_n)
            counts = vec.fit_transform(cluster_texts)
            themes[cluster_id] = list(vec.get_feature_names_out())
        return themes

    def project_2d(self) -> np.ndarray:
        """Reduces embeddings to 2D for plotting in the Streamlit app."""
        if self._embeddings is None:
            raise RuntimeError("Call .fit() first")
        return PCA(n_components=2, random_state=42).fit_transform(self._embeddings)


if __name__ == "__main__":
    sample_texts = [
        "Delivery was very late and the box was crushed.",
        "Package took forever to arrive, quite disappointing.",
        "Amazing build quality, feels very premium.",
        "The material feels cheap and flimsy.",
        "Way overpriced for what you actually get.",
        "Good value for the price, would buy again.",
    ]
    clusterer = ReviewClusterer(n_clusters=3)
    labels = clusterer.fit(sample_texts)
    print("Cluster labels:", labels)
    print("Top terms per cluster:", clusterer.top_terms_per_cluster(sample_texts, labels))