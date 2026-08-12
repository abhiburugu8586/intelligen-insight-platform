"""
Advanced feature: Explainable AI (XAI) using SHAP.

Explains predictions of the TfidfCategoryClassifier (see classification.py):
- Global explanation: which words matter most across all predictions
- Local explanation: for a single review, which words pushed it toward its
  predicted category

Owner: Person B

Note: SHAP works most cleanly with the TF-IDF + LogisticRegression
classifier (TfidfCategoryClassifier), not the zero-shot transformer
pipeline, since it needs access to model internals / a well-defined
feature space. If you want to explain a transformer instead, shap.Explainer
also supports HuggingFace pipelines directly (see the commented example
at the bottom) but it is slower and heavier to run.
"""

from __future__ import annotations
from typing import List
import shap

from src.classification import TfidfCategoryClassifier


class ClassifierExplainer:
    def __init__(self, classifier: TfidfCategoryClassifier, background_texts: List[str]):
        """
        Args:
            classifier: a *fitted* TfidfCategoryClassifier
            background_texts: a sample of training texts used as SHAP's
                background distribution (a subset, e.g. 50-100 examples, is
                usually enough and keeps this fast).
        """
        if not classifier.is_fitted:
            raise RuntimeError("Fit the classifier before creating an explainer")

        self.classifier = classifier
        # shap.Explainer auto-picks a suitable algorithm; for a sklearn
        # Pipeline wrapping text input, we wrap predict_proba directly.
        self._explainer = shap.Explainer(
            self.classifier.model.predict_proba,
            shap.maskers.Text(),
        )

    def explain(self, texts: List[str]):
        """Returns a shap.Explanation object for the given texts."""
        return self._explainer(texts)

    def explain_and_plot_local(self, text: str, out_path: str | None = None):
        """
        Produces a local (per-review) explanation. In a notebook this
        renders inline; in a script pass out_path to save as HTML.
        """
        shap_values = self.explain([text])
        plot = shap.plots.text(shap_values, display=False)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(plot)
        return plot


if __name__ == "__main__":
    # Quick manual test — mirrors the toy example in classification.py
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

    explainer = ClassifierExplainer(clf, background_texts=texts)
    explainer.explain_and_plot_local(
        "My order never showed up on time.",
        out_path="shap_explanation_example.html",
    )
    print("Saved example SHAP explanation to shap_explanation_example.html")

# --- Alternative: explaining a transformer pipeline instead of sklearn ---
# from transformers import pipeline
# sentiment_pipe = pipeline("sentiment-analysis", return_all_scores=True)
# explainer = shap.Explainer(sentiment_pipe)
# shap_values = explainer(["This product is amazing!"])
# shap.plots.text(shap_values)
