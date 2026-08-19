"""
IntelliGen Customer Insight & Response Platform — demo UI.

Run with:
    streamlit run app/streamlit_app.py
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
    st.caption(
        "Clustering groups reviews by semantic similarity. On this sample "
        "dataset (which spans multiple product types), clusters primarily "
        "reflect **product category** (e.g. books, movies, toys) rather than "
        "complaint type — a genuine and useful finding in its own right, "
        "since it shows the platform can automatically segment mixed "
        "feedback by product line without any manual tagging. On a "
        "single-category dataset (e.g. one retailer's product reviews), "
        "the same method would be expected to surface complaint-type themes "
        "instead."
    )
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
    category_counts = df["category"].value_counts()
    top_category = category_counts.index[0] if len(category_counts) > 0 else "N/A"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Negative sentiment reviews", f"{negative_pct}%")
    with col2:
        st.metric("Reviews analysed", f"{len(df)}")
    with col3:
        st.metric("Most common issue", top_category)

    st.markdown("### The problem this solves for IntelliGen's clients")
    st.markdown(
        """
        Retail and e-commerce clients receive large volumes of customer feedback
        every day, but manually reading and routing each review to the right team
        is slow, inconsistent, and doesn't scale. Important complaints can sit
        unnoticed for days while staff work through a backlog, and emerging
        problems (e.g. a batch of damaged stock, a confusing pricing change)
        often aren't spotted until many customers have already been affected.
        """
    )

    st.markdown("### How each component adds value")
    st.markdown(
        f"""
        - **Sentiment analysis** automatically flags negative reviews for
          priority attention, instead of staff reading every review in order.
          In this sample, **{negative_pct}%** of reviews were negative and
          would be surfaced first.
        - **Classification** routes each review to the right internal team
          automatically (e.g. delivery issues to logistics, pricing complaints
          to the commercial team) rather than requiring manual triage. The most
          common issue in this sample was **{top_category}**, which a client
          could act on directly.
        - **Clustering** automatically segments mixed customer feedback into
          coherent groups without any manual tagging. On this sample dataset
          (which spans multiple product types), clusters primarily reflect
          product category — demonstrating the platform's ability to organise
          large volumes of unlabelled feedback. On a single-category dataset,
          the same technique would be expected to surface complaint-type
          themes instead, giving early visibility into developing issues.
        - **Explainable AI (SHAP)** makes every automated decision auditable:
          staff can see exactly which words drove a classification, which
          supports trust, quality control, and compliance conversations with
          clients who need to justify automated decisions to regulators or
          their own customers.
        """
    )

    st.markdown("### Why this matters commercially for IntelliGen")
    st.markdown(
        """
        This kind of tool lets IntelliGen offer clients faster response times to
        customer complaints, better visibility into recurring product or service
        issues, and a defensible, explainable process for how customer feedback
        is triaged — a meaningful differentiator versus a generic sentiment
        dashboard that only classifies but can't explain its own decisions.
        """
    )

    st.caption(
        "Figures above are calculated live from the currently loaded sample "
        "and will update automatically based on the reviews analysed."
    )