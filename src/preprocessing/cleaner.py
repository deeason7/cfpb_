# src/preprocessing/cleaner.py

import pandas as pd
import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Perform advanced NLP cleaning on complaint narratives.
    """
    if pd.isnull(text):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Lowercase
    text = text.lower()

    # Remove HTML, URLs, emails, and numbers
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove special characters and extra spaces
    text = re.sub(r"[^a-z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()

    return text

def clean_complaints_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the CFPB complaints dataset:
    - Drop unusable and highly null columns
    - Remove duplicates
    - Clean narrative
    """
    # Drop rows with missing complaint text
    df = df.dropna(subset=["consumer_complaint_narrative"])

    # Drop high-null or irrelevant columns
    drop_cols = ["tags", "company_public_response", "consumer_consent_provided"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # Drop duplicates by complaint ID
    df = df.drop_duplicates(subset=["complaint_id"])

    # Clean the narrative
    df["text_cleaned"] = df["consumer_complaint_narrative"].apply(clean_text)

    # Optional: trim whitespace on object columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    return df
