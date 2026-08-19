"""
Sentiment analysis: classifies reviews as positive or negative using a
pretrained transformer model.
"""

from __future__ import annotations
from typing import List, Dict
from transformers import pipeline


class SentimentAnalyzer:
    """Wraps a pretrained sentiment-analysis transformer pipeline."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self._pipe = pipeline("sentiment-analysis", model=model_name)

    def predict(self, texts: List[str]) -> List[Dict]:
        """Returns a list of {"label": "POSITIVE"/"NEGATIVE", "score": float}."""
        return self._pipe(texts, truncation=True)

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample = [
        "This product completely broke after two days, terrible quality.",
        "Fast delivery and works exactly as described, very happy!",
    ]
    for text, result in zip(sample, analyzer.predict(sample)):
        print(f"{result['label']} ({result['score']:.2f}) -- {text}")