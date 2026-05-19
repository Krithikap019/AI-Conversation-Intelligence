from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from src.models import get_embedding_model

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass
class VectorStore:
    index: faiss.IndexFlatL2
    texts: List[str]
    embeddings: np.ndarray


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return chunks


def build_vector_store(texts: List[str]) -> VectorStore:
    model = get_embedding_model()

    chunks: List[str] = []
    for text in texts:
        if str(text).strip():
            chunks.extend(chunk_text(str(text)))

    if not chunks:
        raise ValueError("No text available to build vector store.")

    embeddings = model.encode(chunks, show_progress_bar=False).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return VectorStore(index=index, texts=chunks, embeddings=embeddings)


def retrieve(question: str, store: VectorStore, top_k: int = 4) -> List[str]:
    model = get_embedding_model()
    q_emb = model.encode([question], show_progress_bar=False).astype("float32")
    _, idx = store.index.search(q_emb, top_k)

    return [store.texts[i] for i in idx[0] if 0 <= i < len(store.texts)]


def _extract_keywords(question: str) -> List[str]:
    q = question.lower()

    stopwords = {
        "which", "what", "who", "where", "when", "why", "how",
        "the", "a", "an", "do", "does", "did", "is", "are", "was", "were",
        "can", "could", "would", "should", "about", "that", "this", "these",
        "those", "mention", "mentions", "mentioned", "conversation", "conversations",
        "issue", "issues", "look", "looks", "like", "tell", "me"
    }

    tokens = re.findall(r"[a-zA-Z]+", q)
    keywords = [t for t in tokens if t not in stopwords and len(t) > 2]

    return keywords


def _local_answer(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "No relevant conversations were retrieved."

    q = question.lower()
    keywords = _extract_keywords(question)

    matched_contexts = []
    if keywords:
        for c in contexts:
            c_lower = c.lower()
            if any(k in c_lower for k in keywords):
                matched_contexts.append(c)

    if not matched_contexts:
        matched_contexts = contexts

    if any(phrase in q for phrase in ["which conversations", "mention", "mentions"]):
        lines = []
        for i, c in enumerate(matched_contexts, 1):
            lines.append(f"{i}. {c}")
        return "These retrieved conversations appear most relevant:\n\n" + "\n".join(lines)

    if any(phrase in q for phrase in ["common complaints", "most common", "recurring", "themes", "trend", "trends"]):
        counts = {
            "billing/refund issues": 0,
            "technical/access issues": 0,
            "shipping/delivery issues": 0,
            "account/subscription issues": 0,
            "invoice/reporting questions": 0,
        }

        for c in contexts:
            c_lower = c.lower()
            if any(x in c_lower for x in ["refund", "charged", "billed", "invoice", "payment"]):
                counts["billing/refund issues"] += 1
            if any(x in c_lower for x in ["crashing", "cannot access", "reset password", "login", "dashboard"]):
                counts["technical/access issues"] += 1
            if any(x in c_lower for x in ["delivery", "package", "arrived", "shipping"]):
                counts["shipping/delivery issues"] += 1
            if any(x in c_lower for x in ["subscription", "account is still active", "canceled"]):
                counts["account/subscription issues"] += 1
            if any(x in c_lower for x in ["invoice", "expense reporting"]):
                counts["invoice/reporting questions"] += 1

        ranked = [(k, v) for k, v in counts.items() if v > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if not ranked:
            return "I retrieved relevant conversations, but no clear recurring theme was detected."

        lines = [f"- {theme}: {count} matching conversation(s)" for theme, count in ranked]
        return "Main patterns in the retrieved conversations:\n\n" + "\n".join(lines)

    if any(phrase in q for phrase in ["urgent", "escalate", "high priority"]):
        urgent_hits = [
            c for c in contexts
            if any(x in c.lower() for x in ["urgent", "as soon as possible", "deadline", "escalate", "cannot access"])
        ]

        if urgent_hits:
            lines = [f"{i}. {c}" for i, c in enumerate(urgent_hits, 1)]
            return "These conversations look urgent based on the retrieved context:\n\n" + "\n".join(lines)

        return "No strongly urgent conversation was identified in the retrieved context."

    lines = [f"{i}. {c}" for i, c in enumerate(matched_contexts[:3], 1)]
    return "Here are the most relevant retrieved conversations:\n\n" + "\n".join(lines)


def _answer_with_openai(question: str, contexts: List[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return _local_answer(question, contexts)

    client = OpenAI(api_key=api_key)

    context_block = "\n\n".join(f"Source {i+1}: {c}" for i, c in enumerate(contexts))

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=(
            "You are a customer conversation analyst. "
            "Answer using only the provided context. "
            "If the user asks which conversations mention something, list the relevant source texts clearly. "
            "If the answer is not supported by the context, say that clearly.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_block}"
        ),
        temperature=0,
    )

    return response.output_text.strip()


def answer_question(
    question: str,
    store: VectorStore,
    use_openai: bool = False,
) -> Tuple[str, List[str]]:
    contexts = retrieve(question, store, top_k=4)

    if use_openai:
        answer = _answer_with_openai(question, contexts)
    else:
        answer = _local_answer(question, contexts)

    return answer, contexts