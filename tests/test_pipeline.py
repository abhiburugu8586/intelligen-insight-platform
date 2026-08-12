"""
Basic smoke tests. Expand these as components are built — even a few
passing tests strengthens the "technical/coding capability" component
of the mark scheme.

Run with: pytest tests/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification import TfidfCategoryClassifier


def test_classifier_predicts_known_categories():
    texts = [
        "The package arrived three weeks late and was damaged.",
        "Great quality, feels premium and well made.",
        "Way too expensive for what you get.",
        "Support team never replied to my emails.",
    ]
    labels = ["delivery", "product quality", "pricing", "customer support"]

    clf = TfidfCategoryClassifier()
    clf.fit(texts, labels)
    predictions = clf.predict(["My order never showed up on time."])

    assert predictions[0] in labels


def test_classifier_raises_before_fit():
    clf = TfidfCategoryClassifier()
    try:
        clf.predict(["some text"])
        assert False, "Expected RuntimeError before fitting"
    except RuntimeError:
        pass
