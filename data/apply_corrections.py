"""
Applies a specific set of manually-reviewed label corrections to the
training set, identified from reviewing the top disagreements flagged by
find_label_errors.py.

Usage:
    python data/apply_corrections.py
"""

import pandas as pd

# Each entry: a unique substring to find the row, and its corrected category.
# These were manually reviewed - mostly book/movie/music reviews that got
# auto-labelled into a category due to a coincidental keyword match.
CORRECTIONS = [
    ("worth the time and effort to read", "other"),
    ("This soundtrack is my favorite music", "other"),
    ("This was my first Patricia Cornwell novel", "other"),
    ("a man who has just been killed by the electric chair", "other"),
    ("the most important holiday every year is Academy Awards", "other"),
    ("a great book if you are not familiar with management accounting", "other"),
    ("disappointed that this wasn't a longer book", "other"),
    ("This movie was not an original concept", "other"),
    ("incorporates a lot of the favorites from other sets", "other"),
    ("I am a figure skating fan and found the Barbie Gymnastics", "other"),
    ("This pepper mill does not work well at all", "product quality"),
    ("I have been using this product for 5+ years", "product quality"),
]


def main():
    df = pd.read_csv("data/category_training_set.csv")
    applied = 0

    for substring, new_category in CORRECTIONS:
        mask = df["text"].str.contains(substring, regex=False, na=False)
        matched = mask.sum()
        if matched == 0:
            print(f"WARNING: no match found for: '{substring[:50]}...'")
            continue
        if matched > 1:
            print(f"WARNING: multiple matches ({matched}) for: '{substring[:50]}...' - skipping to avoid wrong edits")
            continue
        df.loc[mask, "category"] = new_category
        applied += 1

    df.to_csv("data/category_training_set.csv", index=False)
    print(f"\nApplied {applied} corrections out of {len(CORRECTIONS)}")
    print("\nUpdated category distribution:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()