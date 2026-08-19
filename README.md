# IntelliGen Customer Insight & Response Platform

A prototype AI platform built for IntelliGen, an AI/ML consultancy, that analyses
customer reviews to surface sentiment, categorise feedback, discover emerging
themes, and explain *why* the model made each prediction.

## What it does

| Module | AI/ML Topic | Description |
|---|---|---|
| `src/sentiment.py` | Sentiment Analysis | Classifies reviews as positive / negative |
| `src/classification.py` | Classification | Tags reviews by category (delivery, quality, pricing, support, other) |
| `src/clustering.py` | Clustering | Groups reviews into unsupervised themes using embeddings + K-Means (k=7, selected via silhouette score analysis) |
| `src/advanced_feature.py` | Explainable AI (XAI) | Uses SHAP to explain the classification model's predictions, globally and per-review |
| `src/pipeline.py` | — | Wires the above together into a single pipeline |
| `app/streamlit_app.py` | — | Interactive demo UI |

## Team & workflow

- **Colab** — used for model experimentation and prototyping (see `notebooks/`)
- **VS Code** — used for building the integrated pipeline, the Streamlit app,
  and all git commits

Workflow: prototype a component in Colab, then build and integrate the final
version in `src/` using VS Code.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

```bash
python data/download_data.py --n 500
```

This downloads a sample of the [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
dataset and saves it to `data/sample_reviews.csv`.

## Building the classification training set

The category classifier (`src/classification.py`) is trained on hand-labelled
data. To rebuild this from scratch:

```bash
python data/build_training_pool.py      # samples and auto-labels candidate reviews
# review/correct data/to_label.csv manually, save as data/category_training_set.csv
python data/find_label_errors.py        # flags likely mislabelled rows for review
python -m src.train_classifier          # trains and evaluates the classifier
```

The current `data/category_training_set.csv` (125 examples) already reflects
this process, including manual review and correction.

## Running the app

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Running the pipeline directly (no UI)

```bash
python -m src.pipeline --input data/sample_reviews.csv --output data/results.csv
```

## Expected output

Running the pipeline or the app on a batch of reviews produces:
- a sentiment label + confidence per review
- a category label per review
- a cluster ID + theme description per review (7 clusters)
- a SHAP explanation (highlighted contributing words) for any selected review

## Advanced feature: Explainable AI (SHAP)

XAI was chosen as our advanced feature (not covered in detail in the module).
SHAP is applied to the classification model to show which words in a review
drove its predicted category, both globally (across all predictions) and
locally (for a single review). See `data/shap_examples/` for generated
example explanations. This directly supports the ethics/transparency section
of the report — explainability is a key requirement for responsible,
accountable AI systems.

## Key findings

- The classifier achieves ~52% accuracy on held-out data (vs. a 20% random
  baseline across 5 categories), trained on 125 hand-labelled examples — a
  reasonable result for a proof-of-concept, with clear scope for improvement
  given more labelled data (see `docs/ethics_and_business.md` for full
  discussion).
- Clustering (k=7, selected via silhouette score) on this general-purpose
  review sample groups reviews primarily by product category rather than
  complaint type, since the sample spans multiple product types. This is a
  genuine and useful finding — see `docs/ethics_and_business.md` for the
  full explanation.

## Team contributions

| Name | Student number | Component(s) |
|---|---|---|
| Abhishekar | [35053754] | Sentiment analysis, classification, training data, GitHub setup |
| EkpezuEgwu | [34059265] | Clustering, advanced feature (XAI / SHAP) |
| Geetanjali | [35046918] | Integration, Streamlit app, business benefits & ethics/legal/environmental review |

## AI tool usage / AITS statement

See `docs/aits_statement.md` — completed individually by each team member
per the module's AI Transparency Statement requirement.