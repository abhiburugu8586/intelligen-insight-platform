"""
Downloads a small sample of the Amazon Polarity review dataset from
HuggingFace and saves it locally as a CSV so the rest of the pipeline
doesn't need network access every run.

Usage:
    python data/download_data.py --n 500
"""

import argparse
import pandas as pd
from datasets import load_dataset


def download_sample(n: int = 500, out_path: str = "data/sample_reviews.csv") -> None:
    print(f"Loading amazon_polarity dataset (streaming first {n} rows)...")
    ds = load_dataset("amazon_polarity", split="train", streaming=True)

    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append(
            {
                "review_id": i,
                "title": row["title"],
                "text": row["content"],
                # amazon_polarity label: 0 = negative, 1 = positive
                "true_sentiment": "positive" if row["label"] == 1 else "negative",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} reviews to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of reviews to sample")
    parser.add_argument("--out", type=str, default="data/sample_reviews.csv")
    args = parser.parse_args()

    download_sample(n=args.n, out_path=args.out)
