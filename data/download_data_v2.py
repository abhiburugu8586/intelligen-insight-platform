"""
Downloads the Twitter US Airline Sentiment dataset, which comes with
pre-labelled complaint reasons, and maps those onto our project's category
scheme (delivery / product quality / pricing / customer support / other).

This replaces manual hand-labelling: the dataset already tells us the
reason behind each negative review, so we just need to map its labels
onto ours.

Usage:
    python data/download_data_v2.py --n 500
"""

import argparse
import pandas as pd
from datasets import load_dataset

# Maps the dataset's own "negativereason" labels onto our category scheme.
REASON_TO_CATEGORY = {
    "Late Flight": "delivery",
    "Cancelled Flight": "delivery",
    "Flight Attendant Complaints": "customer support",
    "Bad Flight": "product quality",
    "Damaged Luggage": "product quality",
    "Lost Luggage": "product quality",
    "Flight Booking Problems": "pricing",
    "Customer Service Issue": "customer support",
    "longlines": "customer support",
    "Can't Tell": "other",
}


def download_sample(n: int = 500, out_path: str = "data/sample_reviews.csv") -> None:
    print(f"Loading Twitter US Airline Sentiment dataset...")
    ds = load_dataset("mattbit/tweet-sentiment-airlines", split="train")

    df = ds.to_pandas()

    # Keep only rows that have a review reason (mostly negative tweets);
    # positive/neutral tweets won't have a category since they weren't
    # asked "what went wrong."
    df = df.dropna(subset=["text"])

    # Map dataset's own sentiment label to our sentiment format
    df["true_sentiment"] = df["airline_sentiment"].str.lower()

    # Map complaint reason to our category scheme; anything without a
    # reason (positive/neutral tweets) becomes "other"
    df["category"] = df["negativereason"].map(REASON_TO_CATEGORY).fillna("other")

    df = df[["text", "true_sentiment", "category"]].reset_index(drop=True)
    df.insert(0, "review_id", range(len(df)))

    sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    sample.to_csv(out_path, index=False)

    print(f"Saved {len(sample)} reviews to {out_path}")
    print("\nCategory distribution:")
    print(sample["category"].value_counts())
    print("\nSentiment distribution:")
    print(sample["true_sentiment"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of reviews to sample")
    parser.add_argument("--out", type=str, default="data/sample_reviews.csv")
    args = parser.parse_args()

    download_sample(n=args.n, out_path=args.out)