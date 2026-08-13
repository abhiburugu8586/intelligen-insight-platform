"""
Finds rows in the training set that are likely mislabeled, so you can
review a short, targeted list instead of re-checking all 125 rows.

How it works: trains the classifier on ALL your labelled data, then asks
it to re-predict every row it was trained on. Rows where the model is
both (a) wrong and (b) confident in its own (different) answer are the
most likely to be genuine label mistakes -- a real, low-effort way to
catch data entry issues (e.g. "pricing" applied to a review that isn't
about price at all).

Usage:
    python data/find_label_errors.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import cross_val_predict
from src.classification import TfidfCategoryClassifier

df = pd.read_csv("data/category_training_set.csv")
df = df[df["category"].notna() & (df["category"] != "")].reset_index(drop=True)

clf = TfidfCategoryClassifier()

# IMPORTANT: use cross-validated out-of-fold predictions, not the same
# model re-predicting the data it was trained on. With only ~125 examples,
# a same-data fit/predict essentially memorizes the training labels
# (overfitting), which hides real label errors instead of catching them.
# cross_val_predict holds each row out of training when predicting it,
# giving a much more honest signal of which labels don't fit the pattern.
probs = cross_val_predict(clf.model, df["text"], df["category"], cv=5, method="predict_proba")

# fit once on full data to read out the class order (label names only,
# not used for the cross-validated predictions above)
clf.fit(df["text"].tolist(), df["category"].tolist())
classes = clf.model.named_steps["clf"].classes_

predicted = probs.argmax(axis=1)
confidence = probs.max(axis=1)

df["predicted"] = [classes[i] for i in predicted]
df["confidence"] = confidence
df["mislabeled_or_hard"] = df["predicted"] != df["category"]

suspects = df[df["mislabeled_or_hard"]].sort_values("confidence", ascending=False)

print(f"Found {len(suspects)} out of {len(df)} rows where the model disagrees with your label.")
print("Showing the top 25, sorted by how confident the model was in its own answer")
print("(most confident disagreements are usually the most likely genuine mistakes):\n")

for _, row in suspects.head(25).iterrows():
    print(f"  current label: {row['category']:20s} model suggests: {row['predicted']:20s} (conf {row['confidence']:.2f})")
    print(f"  text: {row['text'][:100]}")
    print()

suspects[["text", "category", "predicted", "confidence"]].to_csv("data/label_review_needed.csv", index=False)
print(f"Saved all {len(suspects)} disagreements to data/label_review_needed.csv for editing")
print("You don't need to review all of them -- focus on the highest-confidence ones first,")
print("since those are most likely to be genuine label mistakes rather than just hard cases.")