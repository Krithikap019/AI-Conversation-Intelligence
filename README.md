# AI Conversation Intelligence Platform

A Streamlit app that analyzes chats, call transcripts, tickets, emails, or feedback using:
- Hugging Face transformers
- sentence-transformer embeddings
- clustering
- summarization
- named entity recognition
- RAG with FAISS

## Features
- Zero-shot intent classification
- Sentiment analysis
- NER / entity extraction
- Urgency scoring
- Theme clustering with embeddings
- Corpus summarization
- RAG Q&A over uploaded conversations

## Project structure

```text
conversation_intelligence/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   └── sample_conversations.csv
└── src/
    ├── analytics.py
    ├── models.py
    ├── preprocess.py
    └── rag.py
```

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Cloud
1. Push this folder to GitHub.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the entrypoint to `app.py`.
4. Optional: add `OPENAI_API_KEY` in Secrets if you want stronger RAG answers.

## Recommended interview talking points
- Why zero-shot classification was used for rapid iteration
- Why embeddings + clustering help with unlabeled data
- How FAISS retrieval grounds generation and reduces hallucinations
- How you would evaluate intent quality, retrieval quality, and answer grounding
- Error analysis on long transcripts, duplicates, and noisy labels

## Example interview answer

“I built a conversation intelligence app that takes in support chats or transcripts and turns them into structured insights. I used Hugging Face transformers for zero-shot intent classification, sentiment analysis, and named entity recognition. For unlabeled issue discovery, I embedded conversations with `all-MiniLM-L6-v2` and clustered them to surface recurring themes. I also added a RAG layer using FAISS so the user can ask natural-language questions and get grounded answers over the uploaded transcripts. The whole app is Streamlit-based so it is simple to demo, but the components are production-friendly and can be moved behind an API later.”

## Notes
- This is a practical MVP. For a stronger production system, move model inference to a FastAPI backend, store vectors in a persistent database, add auth, and add proper evaluation dashboards.
