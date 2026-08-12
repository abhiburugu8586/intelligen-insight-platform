"""
Sentiment Analysis module.

Uses a pretrained HuggingFace transformer to classify each review as
positive, negative (and optionally neutral, if you switch models/thresholds).

Owner: Person A
"""

from __future__ import annotations
from typing import List, Dict
from transformers import pipeline


class SentimentAnalyzer:
    """Wraps a pretrained sentiment-analysis transformer pipeline."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # Loads a lightweight, well-known sentiment model.
        # Swap model_name for something else (e.g. a 3-class model) if you
        # want neutral as an explicit label rather than a confidence threshold.
        self._pipe = pipeline("sentiment-analysis", model=model_name)

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Args:
            texts: list of review strings.

        Returns:
            list of dicts: {"label": "POSITIVE"/"NEGATIVE", "score": float}
        """
        # transformers pipelines truncate long inputs automatically when
        # truncation=True is passed; add that if you hit token-length errors.
        return self._pipe(texts, truncation=True)

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]


if __name__ == "__main__":
    # Quick manual test
    analyzer = SentimentAnalyzer()
    sample = [
        "This product completely broke after two days, terrible quality.",
        "Fast delivery and works exactly as described, very happy!",
    ]
    for text, result in zip(sample, analyzer.predict(sample)):
        print(f"{result['label']} ({result['score']:.2f}) -- {text}")
