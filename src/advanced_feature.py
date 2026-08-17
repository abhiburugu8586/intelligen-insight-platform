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
    import os
    import pandas as pd

    # Use the real, hand-labelled training data instead of a toy example set,
    # so explanations reflect the actual model used in the app.
    training_path = "data/category_training_set.csv"
    if os.path.exists(training_path):
        df = pd.read_csv(training_path)
        df = df[df["category"].notna() & (df["category"] != "")]
        texts = df["text"].tolist()
        labels = df["category"].tolist()
        print(f"Training on {len(texts)} real labelled examples from {training_path}")
    else:
        # Fallback toy set only used if the real training data isn't present
        texts = [
            "The package arrived three weeks late and was damaged.",
            "Great quality, feels premium and well made.",
            "Way too expensive for what you get.",
            "Support team never replied to my emails.",
        ]
        labels = ["delivery", "product quality", "pricing", "customer support"]
        print("WARNING: data/category_training_set.csv not found, using toy fallback data")

    clf = TfidfCategoryClassifier()
    clf.fit(texts, labels)

    explainer = ClassifierExplainer(clf, background_texts=texts)

    # Generate explanations for a few different example reviews covering
    # different categories, so we have several ready-made outputs for the
    # video/report rather than just one.
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

# --- Alternative: explaining a transformer pipeline instead of sklearn ---
# from transformers import pipeline
# sentiment_pipe = pipeline("sentiment-analysis", return_all_scores=True)
# explainer = shap.Explainer(sentiment_pipe)
# shap_values = explainer(["This product is amazing!"])
# shap.plots.text(shap_values)