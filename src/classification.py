"""
Classification module.

Tags each review with a category (e.g. delivery, quality, pricing, support,
other). Two approaches are provided:

1. ZeroShotCategoryClassifier - no training data needed, uses a pretrained
   zero-shot model. Good for getting something working immediately.
2. TfidfCategoryClassifier - a simple trainable scikit-learn classifier.
   Use this if you have (or create) a small labelled training set and want
   a "real" trained model to explain with SHAP later.

Owner: Person A
"""

from __future__ import annotations
from typing import List, Dict, Sequence
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEFAULT_CATEGORIES = ["delivery", "product quality", "pricing", "customer support", "other"]


class ZeroShotCategoryClassifier:
    """No training required - classifies text against a set of category labels."""

    def __init__(self, categories: Sequence[str] = DEFAULT_CATEGORIES,
                 model_name: str = "facebook/bart-large-mnli"):
        self.categories = list(categories)
        self._pipe = pipeline("zero-shot-classification", model=model_name)

    def predict(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            out = self._pipe(text, candidate_labels=self.categories)
            results.append({"label": out["labels"][0], "score": out["scores"][0]})
        return results


class TfidfCategoryClassifier:
    """
    A trainable, explainable classifier (TF-IDF + Logistic Regression).
    Recommended for use with the SHAP advanced feature, since SHAP works
    cleanly with scikit-learn models.
    """

    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> None:
        self.model.fit(texts, labels)
        self.is_fitted = True

    def predict(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return list(self.model.predict(texts))

    def predict_proba(self, texts: List[str]):
        return self.model.predict_proba(texts)


if __name__ == "__main__":
    # Quick manual test with a tiny toy training set.
    # Replace with real (weak/self-labelled) training data - e.g. label a
    # few hundred reviews by keyword rules or manually as a starting point.
    texts = [
        "The package arrived three weeks late and was damaged.",
        "Great quality, feels premium and well made.",
        "Way too expensive for what you get.",
        "Support team never replied to my emails.",
        "Shipping was delayed again, very frustrating.",
        "Excellent build quality, exceeded expectations.",
    ]
    labels = ["delivery", "product quality", "pricing", "customer support", "delivery", "product quality"]

    clf = TfidfCategoryClassifier()
    clf.fit(texts, labels)
    print(clf.predict(["My order never showed up on time."]))
