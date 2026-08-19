"""
Classification: tags each review with a category (delivery, product
quality, pricing, customer support, other).

Two classifiers are provided - a zero-shot model (no training needed) and
a trainable TF-IDF + Logistic Regression model (used for the SHAP
explainability feature, since it exposes model internals SHAP can work with).
"""

from __future__ import annotations
from typing import List, Dict, Sequence
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DEFAULT_CATEGORIES = ["delivery", "product quality", "pricing", "customer support", "other"]


class ZeroShotCategoryClassifier:
    """Classifies text against a set of category labels with no training required."""

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
    """Trainable TF-IDF + Logistic Regression category classifier."""

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