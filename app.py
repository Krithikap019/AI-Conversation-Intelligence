import os
from io import StringIO
from typing import List

import pandas as pd
import streamlit as st

from src.analytics import (
    analyze_transcripts,
    cluster_transcripts,
    summarize_corpus,
)
from src.preprocess import load_uploaded_data
from src.rag import build_vector_store, answer_question

st.set_page_config(
    page_title="AI Conversation Intelligence Platform",
    page_icon="💬",
    layout="wide",
)

st.title("💬 AI Conversation Intelligence Platform")
st.caption(
    "Analyze chats, support tickets, emails, or call transcripts with transformers, embeddings, clustering, and RAG."
)

with st.sidebar:
    st.header("Settings")
    default_labels = [
        "billing issue",
        "technical problem",
        "refund request",
        "cancel subscription",
        "product question",
        "shipping delay",
        "complaint",
        "praise",
    ]
    labels_text = st.text_area(
        "Intent labels (one per line)",
        value="\n".join(default_labels),
        height=180,
        help="Used by zero-shot intent classification.",
    )
    max_rows = st.slider("Max rows to analyze", 10, 250, 50, step=10)
    n_clusters = st.slider("Clusters", 2, 8, 4)
    use_openai = st.toggle(
        "Use OpenAI for RAG answer (optional)",
        value=False,
        help="Turn on only if OPENAI_API_KEY is set in Streamlit secrets or environment variables.",
    )
    st.markdown("---")
    st.write("Accepted files: CSV, TXT")

uploaded_file = st.file_uploader(
    "Upload a CSV or TXT file",
    type=["csv", "txt"],
    help="CSV should contain either a `text` column or transcript-like columns such as message, body, transcript, or conversation.",
)

sample_mode = st.checkbox("Use bundled sample data", value=uploaded_file is None)

if uploaded_file is not None:
    df = load_uploaded_data(uploaded_file)
elif sample_mode:
    df = pd.read_csv("data/sample_conversations.csv")
else:
    df = pd.DataFrame(columns=["text"])

if not df.empty:
    df = df.head(max_rows).copy()

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

candidate_labels = [x.strip() for x in labels_text.splitlines() if x.strip()]

if df.empty:
    st.info("Upload a file or enable sample data to start.")
    st.stop()

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "cluster_df" not in st.session_state:
    st.session_state.cluster_df = None
if "corpus_summary" not in st.session_state:
    st.session_state.corpus_summary = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

col_a, col_b, col_c = st.columns(3)
run_analysis = col_a.button("Run NLP Analysis", use_container_width=True)
run_clustering = col_b.button("Run Clustering", use_container_width=True)
build_rag = col_c.button("Build RAG Index", use_container_width=True)

if run_analysis:
    with st.spinner("Running transformer pipelines..."):
        st.session_state.analysis_df = analyze_transcripts(df, candidate_labels)
        try:
            st.session_state.corpus_summary = summarize_corpus(
                st.session_state.analysis_df["text"].tolist()
            )
        except Exception as e:
            st.session_state.corpus_summary = f"Summary unavailable: {e}"
if run_clustering:
    with st.spinner("Creating embeddings and clusters..."):
        st.session_state.cluster_df = cluster_transcripts(df, n_clusters=n_clusters)

if build_rag:
    with st.spinner("Embedding chunks and building vector store..."):
        st.session_state.vector_store = build_vector_store(df["text"].tolist())

analysis_tab, cluster_tab, rag_tab = st.tabs(
    ["Analysis", "Clustering", "RAG Q&A"]
)

with analysis_tab:
    if st.session_state.analysis_df is None:
        st.info("Click **Run NLP Analysis** to generate insights.")
    else:
        analyzed = st.session_state.analysis_df
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Rows analyzed", len(analyzed))
        kpi2.metric(
            "High urgency",
            int((analyzed["urgency_score"] >= 0.7).sum()),
        )
        kpi3.metric("Unique intents", analyzed["predicted_intent"].nunique())
        kpi4.metric(
            "Negative sentiment",
            int((analyzed["sentiment_label"] == "NEGATIVE").sum()),
        )

        st.markdown("### Corpus summary")
        st.write(st.session_state.corpus_summary)

        left, right = st.columns([1, 1])
        with left:
            st.markdown("### Intent distribution")
            intent_counts = analyzed["predicted_intent"].value_counts().reset_index()
            intent_counts.columns = ["intent", "count"]
            st.bar_chart(intent_counts.set_index("intent"))

        with right:
            st.markdown("### Sentiment distribution")
            sent_counts = analyzed["sentiment_label"].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            st.bar_chart(sent_counts.set_index("sentiment"))

        st.markdown("### Detailed results")
        st.dataframe(analyzed, use_container_width=True)

with cluster_tab:
    if st.session_state.cluster_df is None:
        st.info("Click **Run Clustering** to generate clusters.")
    else:
        clustered = st.session_state.cluster_df
        st.markdown("### Cluster assignments")
        st.dataframe(clustered, use_container_width=True)
        st.markdown("### Cluster sizes")
        counts = clustered["cluster"].value_counts().sort_index()
        st.bar_chart(counts)
        st.markdown("### Sample transcripts by cluster")
        for cluster_id in sorted(clustered["cluster"].unique()):
            with st.expander(f"Cluster {cluster_id}"):
                subset = clustered[clustered["cluster"] == cluster_id][
                    ["text", "cluster_label"]
                ].head(5)
                st.dataframe(subset, use_container_width=True)

with rag_tab:
    st.markdown(
        "Ask questions like: `What are the most common complaints?`, `Which conversations mention refunds?`, or `What issues look urgent?`"
    )
    question = st.text_input("Ask a question about the uploaded conversations")
    if question:
        if st.session_state.vector_store is None:
            st.warning("Build the RAG index first.")
        else:
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                answer, sources = answer_question(
                    question,
                    st.session_state.vector_store,
                    use_openai=use_openai,
                )
            st.markdown("### Answer")
            st.write(answer)
            st.markdown("### Retrieved context")
            for i, src in enumerate(sources, start=1):
                st.markdown(f"**Source {i}:** {src}")

