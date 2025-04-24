# src/modeling/sentiment_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split

def weak_sentiment(text: str) -> str:
    """
    Basic weak labeling: returns 'negative', 'neutral', or 'positive' based on simple keyword heuristics.
    """
    negative_keywords = ['not', "n't", 'no', 'never', 'bad', 'poor', 'fail']
    positive_keywords = ['good', 'great', 'excellent', 'satisfied', 'happy']
    lower = text.lower()
    if any(kw in lower for kw in negative_keywords):
        return 'negative'
    if any(kw on lower for kw in positive_keywords):
        return 'positive'
    return 'neutral'

def prepare_data(filepath: str)
    """
    1. Load the raw CSV file
    2. Ensure cleaned text is present
    3. Apply weak labeling rules if 'sentiment' column is missing or incomplete
    4. Map sentiment label to integers
    5. Return stratified train/validation split of text and labels
    """

    df = pd.read_csv(filepath)
    df = df[df.get('text_cleaned').notna()].copy()

    #Apply weak labeling if needed
    if 'sentiment' not in df.columns or df['sentiment'].isna().any():
        df['sentiment'] = df['text_cleaned'].apply(weak_sentiment)

    #Define mapping to integer labels
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['sentiment'].map(label_map)

    return train_test_split(
        X,y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )