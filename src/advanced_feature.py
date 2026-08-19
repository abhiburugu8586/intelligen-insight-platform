"""
Advanced feature: Explainable AI (XAI) using SHAP.

Explains predictions of the TfidfCategoryClassifier (see classification.py)
by showing which words in a review pushed the model toward its predicted
category.
"""

from __future__ import annotations
from typing import List
import shap

from src.classification import TfidfCategoryClassifier


class ClassifierExplainer:
    def __init__(self, classifier: TfidfCategoryClassifier, background_texts: List[str]):
        if not classifier.is_fitted:
            raise RuntimeError("Fit the classifier before creating an explainer")

        self.classifier = classifier
        self._explainer = shap.Explainer(
            self.classifier.model.predict_proba,
            shap.maskers.Text(),
        )

    def explain(self, texts: List[str]):
        """Returns a shap.Explanation object for the given texts."""
        return self._explainer(texts)

    def explain_and_plot_local(self, text: str, out_path: str | None = None):
        """Produces a per-review explanation, optionally saved as HTML."""
        shap_values = self.explain([text])
        plot = shap.plots.text(shap_values, display=False)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(plot)
        return plot


if __name__ == "__main__":
    import os
    import pandas as pd

    training_path = "data/category_training_set.csv"
    if os.path.exists(training_path):
        df = pd.read_csv(training_path)
        df = df[df["category"].notna() & (df["category"] != "")]
        texts = df["text"].tolist()
        labels = df["category"].tolist()
        print(f"Training on {len(texts)} labelled examples from {training_path}")
    else:
        texts = [
            "The package arrived three weeks late and was damaged.",
            "Great quality, feels premium and well made.",
            "Way too expensive for what you get.",
            "Support team never replied to my emails.",
        ]
        labels = ["delivery", "product quality", "pricing", "customer support"]
        print("WARNING: data/category_training_set.csv not found, using fallback data")

    clf = TfidfCategoryClassifier()
    clf.fit(texts, labels)

    explainer = ClassifierExplainer(clf, background_texts=texts)

    demo_reviews = [
        "My order never showed up on time and support never replied to my emails.",
        "This is way overpriced for the quality you actually get.",
        "Arrived broken and the box was completely crushed during shipping.",
        "Excellent build quality, definitely worth the price I paid.",
    ]

    os.makedirs("data/shap_examples", exist_ok=True)
    for i, review in enumerate(demo_reviews):
        prediction = clf.predict([review])[0]
        out_path = f"data/shap_examples/example_{i+1}_{prediction.replace(' ', '_')}.html"
        explainer.explain_and_plot_local(review, out_path=out_path)
        print(f"[{prediction}] {review}")
        print(f"  -> saved explanation to {out_path}\n")

    print("Done. Open any of the saved HTML files in a browser to see the SHAP explanation.")