from __future__ import annotations

from typing import Optional

import pandas as pd

TEXT_CANDIDATE_COLUMNS = [
    "text",
    "message",
    "body",
    "conversation",
    "transcript",
    "content",
    "review",
]


def _find_text_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in TEXT_CANDIDATE_COLUMNS:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text_col = _find_text_column(df)
        if text_col is None:
            raise ValueError(
                "CSV must include one of these text columns: "
                + ", ".join(TEXT_CANDIDATE_COLUMNS)
            )
        clean = df[[text_col]].rename(columns={text_col: "text"}).copy()
    elif filename.endswith(".txt"):
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        blocks = [x.strip() for x in raw.split("\n\n") if x.strip()]
        clean = pd.DataFrame({"text": blocks})
    else:
        raise ValueError("Unsupported file type. Use CSV or TXT.")

    clean["text"] = clean["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    clean = clean[clean["text"].str.len() > 0].drop_duplicates().reset_index(drop=True)
    return clean
