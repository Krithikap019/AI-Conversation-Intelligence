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
