"""
Trains the category classifier on the hand-labelled dataset and does a
quick manual sanity check on a few held-out examples.

Usage:
    python -m src.train_classifier
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.classification import TfidfCategoryClassifier


def main():
    df = pd.read_csv("data/category_training_set.csv")
    df = df[df["category"].notna() & (df["category"] != "")]

    print(f"Loaded {len(df)} labelled examples")
    print(df["category"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["category"]
    )

    clf = TfidfCategoryClassifier()
    clf.fit(train_df["text"].tolist(), train_df["category"].tolist())

    predictions = clf.predict(test_df["text"].tolist())
    correct = sum(p == actual for p, actual in zip(predictions, test_df["category"]))
    print(f"\nAccuracy on held-out test set: {correct}/{len(test_df)} = {correct/len(test_df):.1%}")

    print("\nSample predictions:")
    for text, pred, actual in list(zip(test_df["text"], predictions, test_df["category"]))[:10]:
        marker = "correct" if pred == actual else "WRONG"
        print(f"[{marker}] predicted={pred:20s} actual={actual:20s} -- {text[:60]}")


if __name__ == "__main__":
    main()