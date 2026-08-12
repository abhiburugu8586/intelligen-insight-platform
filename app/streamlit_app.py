"""
IntelliGen Customer Insight & Response Platform — demo UI.

Run with:
    streamlit run app/streamlit_app.py

Owner: Person C (integration / UI)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import plotly.express as px

from src.sentiment import SentimentAnalyzer
from src.classification import TfidfCategoryClassifier
from src.clustering import ReviewClusterer
from src.advanced_feature import ClassifierExplainer
from src.pipeline import BOOTSTRAP_TEXTS, BOOTSTRAP_LABELS

st.set_page_config(page_title="IntelliGen Customer Insight Platform", layout="wide")


@st.cache_resource
def load_models():
    sentiment_analyzer = SentimentAnalyzer()
    classifier = TfidfCategoryClassifier()
    classifier.fit(BOOTSTRAP_TEXTS, BOOTSTRAP_LABELS)
    explainer = ClassifierExplainer(classifier, background_texts=BOOTSTRAP_TEXTS)
    return sentiment_analyzer, classifier, explainer


st.title("🔎 IntelliGen Customer Insight & Response Platform")
st.caption("Sentiment · Classification · Clustering · Explainable AI (SHAP)")

sentiment_analyzer, classifier, explainer = load_models()

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload a CSV with a 'text' column", type=["csv"])
manual_text = st.sidebar.text_area("...or paste reviews (one per line)")
n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=4)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif manual_text.strip():
    df = pd.DataFrame({"text": [line for line in manual_text.split("\n") if line.strip()]})
else:
    st.info("Upload a CSV or paste some reviews in the sidebar to get started.")
    st.stop()

texts = df["text"].astype(str).tolist()

with st.spinner("Analysing sentiment..."):
    sentiment_results = sentiment_analyzer.predict(texts)
    df["sentiment"] = [r["label"] for r in sentiment_results]
    df["sentiment_score"] = [round(r["score"], 3) for r in sentiment_results]

with st.spinner("Classifying reviews..."):
    df["category"] = classifier.predict(texts)

with st.spinner("Clustering reviews..."):
    clusterer = ReviewClusterer(n_clusters=min(n_clusters, len(texts)))
    cluster_labels = clusterer.fit(texts)
    df["cluster"] = cluster_labels
    themes = clusterer.top_terms_per_cluster(texts, cluster_labels)

tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🎯 Clusters", "🧠 Explainability (XAI)", "📈 Business Summary"])

with tab1:
    st.subheader("Review-level results")
    st.dataframe(df[["text", "sentiment", "sentiment_score", "category", "cluster"]], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="sentiment", title="Sentiment breakdown"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x="category", title="Category breakdown"), use_container_width=True)

with tab2:
    st.subheader("Discovered themes")
    for cluster_id, terms in themes.items():
        count = int((df["cluster"] == cluster_id).sum())
        st.markdown(f"**Cluster {cluster_id}** ({count} reviews) — top terms: {', '.join(terms)}")

    coords = clusterer.project_2d()
    plot_df = pd.DataFrame(coords, columns=["x", "y"])
    plot_df["cluster"] = df["cluster"].astype(str)
    plot_df["text"] = df["text"]
    st.plotly_chart(
        px.scatter(plot_df, x="x", y="y", color="cluster", hover_data=["text"],
                   title="Review clusters (2D projection)"),
        use_container_width=True,
    )

with tab3:
    st.subheader("Why did the model predict this category?")
    selected_idx = st.selectbox(
        "Choose a review to explain",
        options=range(len(texts)),
        format_func=lambda i: f"[{df['category'][i]}] {texts[i][:80]}...",
    )
    selected_text = texts[selected_idx]
    st.write(f"**Review:** {selected_text}")
    st.write(f"**Predicted category:** {df['category'][selected_idx]}")

    with st.spinner("Computing SHAP explanation..."):
        html_plot = explainer.explain_and_plot_local(selected_text)
        st.components.v1.html(html_plot, height=250, scrolling=True)

    st.caption(
        "Highlighted words show which parts of the review pushed the model "
        "toward (or away from) its predicted category — this is what makes "
        "the model's decisions auditable rather than a black box."
    )

with tab4:
    st.subheader("Business value for IntelliGen's clients")
    negative_pct = round((df["sentiment"] == "NEGATIVE").mean() * 100, 1)
    st.metric("Negative sentiment reviews", f"{negative_pct}%")
    st.markdown(
        """
        - Automatically triage large volumes of customer feedback without manual review
        - Surface emerging complaint themes (via clustering) before they become widespread issues
        - Route reviews to the right team automatically (via classification)
        - Provide auditable, explainable decisions (via SHAP) to support fairness and
          transparency requirements
        """
    )
    st.caption("Replace this section with real figures/analysis for your report and video.")
