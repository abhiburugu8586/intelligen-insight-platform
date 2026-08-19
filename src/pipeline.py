"""
Ties sentiment, classification, clustering, and the SHAP explainer into a
single pipeline that runs end-to-end over a CSV of reviews.

Usage:
    python -m src.pipeline --input data/sample_reviews.csv --output data/results.csv
"""

from __future__ import annotations
import argparse
import os
import pandas as pd

from src.sentiment import SentimentAnalyzer
from src.classification import TfidfCategoryClassifier, DEFAULT_CATEGORIES
from src.clustering import ReviewClusterer
from src.advanced_feature import ClassifierExplainer

_training_path = "data/category_training_set.csv"
if os.path.exists(_training_path):
    _training_df = pd.read_csv(_training_path)
    _training_df = _training_df[_training_df["category"].notna() & (_training_df["category"] != "")]
    BOOTSTRAP_TEXTS = _training_df["text"].tolist()
    BOOTSTRAP_LABELS = _training_df["category"].tolist()
else:
    BOOTSTRAP_TEXTS = [
        "The package arrived three weeks late and was damaged.",
        "Shipping was delayed again, very frustrating.",
        "Great quality, feels premium and well made.",
        "Excellent build quality, exceeded expectations.",
        "Way too expensive for what you get.",
        "Not worth the price at all.",
        "Support team never replied to my emails.",
        "Customer service was unhelpful and slow.",
    ]
    BOOTSTRAP_LABELS = [
        "delivery", "delivery",
        "product quality", "product quality",
        "pricing", "pricing",
        "customer support", "customer support",
    ]


def run_pipeline(input_csv: str, output_csv: str, n_clusters: int = 7) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column")

    texts = df["text"].astype(str).tolist()

    print(f"Running sentiment analysis on {len(texts)} reviews...")
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_results = sentiment_analyzer.predict(texts)
    df["sentiment"] = [r["label"] for r in sentiment_results]
    df["sentiment_score"] = [r["score"] for r in sentiment_results]

    print("Running classification...")
    classifier = TfidfCategoryClassifier()
    classifier.fit(BOOTSTRAP_TEXTS, BOOTSTRAP_LABELS)
    df["category"] = classifier.predict(texts)

    print(f"Running clustering (k={n_clusters})...")
    clusterer = ReviewClusterer(n_clusters=n_clusters)
    cluster_labels = clusterer.fit(texts)
    df["cluster"] = cluster_labels
    themes = clusterer.top_terms_per_cluster(texts, cluster_labels)
    df["cluster_theme"] = df["cluster"].map(lambda c: ", ".join(themes.get(c, [])))

    print("Building SHAP explainer...")
    _ = ClassifierExplainer(classifier, background_texts=BOOTSTRAP_TEXTS)

    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sample_reviews.csv")
    parser.add_argument("--output", type=str, default="data/results.csv")
    parser.add_argument("--clusters", type=int, default=7)
    args = parser.parse_args()

    run_pipeline(args.input, args.output, n_clusters=args.clusters)