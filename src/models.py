from __future__ import annotations

import os

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def get_zero_shot_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )


@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource(show_spinner=False)
def get_ner_pipeline():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )

@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
    )

@st.cache_resource(show_spinner=False)
def get_text2text_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
    )

def get_summarizer2():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        truncation=True,
    )

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
