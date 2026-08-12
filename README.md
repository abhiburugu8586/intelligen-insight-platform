# IntelliGen Customer Insight & Response Platform

A prototype AI platform built for IntelliGen, an AI/ML consultancy, that analyses
customer reviews to surface sentiment, categorise feedback, discover emerging
themes, and explain *why* the model made each prediction.

## What it does

| Module | AI/ML Topic | Description |
|---|---|---|
| `src/sentiment.py` | Sentiment Analysis | Classifies reviews as positive / negative / neutral |
| `src/classification.py` | Classification | Tags reviews by category (delivery, quality, pricing, support, other) |
| `src/clustering.py` | Clustering | Groups reviews into unsupervised themes using embeddings + K-Means |
| `src/advanced_feature.py` | Explainable AI (XAI) | Uses SHAP to explain the classification model's predictions, globally and per-review |
| `src/pipeline.py` | — | Wires the above together into a single pipeline |
| `app/streamlit_app.py` | — | Interactive demo UI |

## Team & workflow

- **Colab** — used for model experimentation, training, and SHAP exploration
  (see `notebooks/`). Free GPU access, fast iteration.
- **VS Code** — used for building the integrated pipeline, the Streamlit app,
  and for all git commits (clean, incremental history).

Workflow: prototype a component in Colab -> once it works, port the cleaned
version into the matching `src/*.py` file in VS Code -> wire it into
`pipeline.py` / the app -> commit.

## Setup (VS Code / local)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Running the pipeline directly (no UI)

```bash
python -m src.pipeline --input data/sample_reviews.csv
```

## Dataset

We use the [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity)
dataset (or substitute Amazon Fine Food Reviews from Kaggle). See
`data/download_data.py` to fetch a sample. Only a small sample is needed for
a working demo — you do not need the full dataset.

## Expected output

Running the pipeline or the app on a batch of reviews produces:
- a sentiment label + confidence per review
- a category label per review
- a cluster ID + short theme description per review
- a SHAP explanation (top contributing words) for each classification

*(Add screenshots of actual output here for the report.)*

## Advanced feature: Explainable AI (SHAP)

We chose XAI (not taught in detail in the module) as our advanced feature.
SHAP is applied to the classification model to show:
- **Global**: which words matter most across all predictions (summary plot)
- **Local**: for a single review, which words pushed it toward its predicted
  category (force / waterfall plot)

This directly supports the ethics/transparency section of the report:
explainability is a key requirement for responsible, accountable AI systems.

## Team contributions

| Name | Student number | Component(s) |
|---|---|---|
| [Person A] | [number] | Sentiment analysis, Classification |
| [Person B] | [number] | Clustering, Advanced feature (XAI) |
| [Person C] | [number] | Integration, Streamlit app, business & ethics writeup |

## AI tool usage / AITS statement

See `docs/aits_statement.md` — to be completed by each member per the
module's AI Transparency Statement requirement.
