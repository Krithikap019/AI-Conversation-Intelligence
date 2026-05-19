from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
from openai import OpenAI

from src.models import (
    get_embedding_model,
    get_ner_pipeline,
    get_sentiment_pipeline,
    get_summarizer2,
    get_zero_shot_classifier,
)


def _truncate(text: str, max_chars: int = 1000) -> str:
    text = str(text)
    return text[:max_chars]


def _compute_urgency(text: str) -> float:
    text_l = text.lower()
    urgent_terms = [
        "urgent",
        "asap",
        "immediately",
        "right away",
        "cancel",
        "angry",
        "frustrated",
        "refund",
        "broken",
        "outage",
        "not working",
        "escalate",
        "deadline",
    ]
    score = sum(term in text_l for term in urgent_terms) / max(len(urgent_terms), 1)
    return float(min(1.0, score * 4))


def analyze_transcripts(df: pd.DataFrame, candidate_labels: List[str]) -> pd.DataFrame:
    classifier = get_zero_shot_classifier()
    sentiment_pipe = get_sentiment_pipeline()
    ner_pipe = get_ner_pipeline()

    rows = []
    for text in df["text"].tolist():
        short_text = _truncate(text, 900)
        intent = classifier(short_text, candidate_labels)
        sentiment = sentiment_pipe(short_text)[0]
        entities = ner_pipe(short_text)
        entities_fmt = ", ".join(
            sorted({f"{ent['word']} ({ent['entity_group']})" for ent in entities})
        )

        rows.append(
            {
                "text": text,
                "predicted_intent": intent["labels"][0],
                "intent_score": round(float(intent["scores"][0]), 4),
                "sentiment_label": sentiment["label"],
                "sentiment_score": round(float(sentiment["score"]), 4),
                "entities": entities_fmt,
                "urgency_score": round(_compute_urgency(text), 4),
            }
        )

    return pd.DataFrame(rows)

# def summarize_corpus(texts):
#     summarizer = get_summarizer2()

#     conversations = [
#         f"Conversation {i+1}: {t}"
#         for i, t in enumerate(texts)
#         if str(t).strip()
#     ]

#     if not conversations:
#         return "No text available to summarize."

#     # Use more of the corpus
#     joined = "\n".join(conversations)[:7000]

#     prompt = f"""
# You are a customer support analyst.

# Analyze the following customer conversations and write a detailed corpus summary.

# Instructions:
# - Write 6 to 8 bullet points.
# - Group similar complaints together.
# - Mention the most common themes.
# - Mention any repeated technical, billing, service, or delivery issues if present.
# - Mention whether the overall sentiment seems negative, neutral, or mixed.
# - Mention any urgent or high-priority patterns.
# - Be specific and descriptive, not just one sentence.

# Conversations:
# {joined}

# Detailed summary:
# """

#     response = summarizer(
#         prompt,
#         max_new_tokens=350,
#         min_new_tokens=120,
#         do_sample=False,
#         num_beams=4
#     )

#     return response[0]["generated_text"].strip()


def summarize_corpus(texts):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "OpenAI API key not set."

    client = OpenAI(api_key=api_key)

    conversations = [
        f"Conversation {i+1}: {t}"
        for i, t in enumerate(texts)
        if str(t).strip()
    ]

    if not conversations:
        return "No text available to summarize."

    joined = "\n".join(conversations)[:3000]

    prompt = f"""
You are a customer support analyst.

Analyze the following conversations and identify the most common customer complaints.
Group similar issues together and summarize the main themes in bullet points.

Conversations:
{joined}

Summary:
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()


def cluster_transcripts(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    model = get_embedding_model()
    texts = df["text"].tolist()
    emb = model.encode(texts, show_progress_bar=False)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(emb)

    out = df.copy()
    out["cluster"] = clusters
    out["cluster_label"] = out.groupby("cluster")["text"].transform(_cluster_keyword_label)
    return out


def _cluster_keyword_label(series: pd.Series) -> str:
    tokens = []
    for text in series.head(10).tolist():
        clean = [
            tok.strip(".,!?()[]{}:;\"'").lower()
            for tok in text.split()
            if len(tok) > 3
        ]
        tokens.extend(clean)
    common = [w for w, _ in Counter(tokens).most_common(3)]
    return ", ".join(common) if common else "misc"
