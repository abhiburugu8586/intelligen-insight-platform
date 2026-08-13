"""
Scans a large pool of Amazon reviews and keeps only ones that strongly
match one of our category keywords, so the resulting file is actually
labellable (rather than 70%+ "other", which is what happens with a small
random sample).

Usage:
    python data/build_training_pool.py
"""

import pandas as pd
from datasets import load_dataset

KEYWORDS = {
    "delivery": [
        "delivery", "shipping", "shipped", "arrived late", "package",
        "packaging", "delayed", "damaged box", "courier", "tracking",
        "took forever to arrive", "never arrived",
    ],
    "product quality": [
        "quality", "broke", "broken", "durable", "durability", "material",
        "cheap", "flimsy", "well made", "sturdy", "defective",
        "stopped working", "fell apart", "poorly made", "well built",
    ],
    "pricing": [
        "price", "expensive", "overpriced", "value for money", "worth the",
        "not worth", "too much money", "discount", "cheap price", "pricey",
        "good value", "waste of money",
    ],
    "customer support": [
        "customer service", "customer support", "refund", "replacement",
        "return policy", "replied to my", "never responded", "helpline",
        "warranty", "complaint", "resolve", "return the item",
    ],
}


def guess_category(text: str):
    """Returns (category, match_count) - match_count used as a confidence signal."""
    text_lower = str(text).lower()
    scores = {}
    for category, keywords in KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    if not scores:
        return None, 0
    best = max(scores, key=scores.get)
    return best, scores[best]


def build_pool(pool_size: int = 5000, target_per_category: int = 25, out_path: str = "data/to_label.csv"):
    print(f"Scanning {pool_size} reviews for confident category matches...")
    ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

    matched_rows = {cat: [] for cat in KEYWORDS}
    other_rows = []

    for i, row in enumerate(ds):
        if i >= pool_size:
            break
        text = row["content"]
        category, score = guess_category(text)

        if category and score >= 1 and len(matched_rows[category]) < target_per_category:
            matched_rows[category].append({"review_id": i, "text": text, "category": category})
        elif category is None and len(other_rows) < target_per_category:
            other_rows.append({"review_id": i, "text": text, "category": "other"})

        # stop early once every bucket is full
        if all(len(v) >= target_per_category for v in matched_rows.values()) and len(other_rows) >= target_per_category:
            break

    all_rows = other_rows[:]
    for cat_rows in matched_rows.values():
        all_rows.extend(cat_rows)

    df = pd.DataFrame(all_rows).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} reviews to {out_path}")
    print("\nCategory distribution (pre-filled suggestions - please review each row):")
    print(df["category"].value_counts())
    print(
        "\nNext: open data/to_label.csv, skim each row, fix any wrong "
        "suggestions. Since these were keyword-matched, most should already "
        "be correct - this is a review pass, not labelling from scratch."
    )


if __name__ == "__main__":
    build_pool()
    