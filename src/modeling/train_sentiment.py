# src/modeling/train_sentiment.py

import os
import pickle
import numpy as np
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification
)


# ─── 1) WEAK‐LABELING ───────────────────────────────────────────────────────────

_analyzer = SentimentIntensityAnalyzer()

def weak_sentiment(text: str) -> str:
    """
    Combine TextBlob polarity with VADER compound score.
    final_score = 0.5 * tb + 0.5 * vader
    thresholds at ±0.05
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "neutral"
    tb = TextBlob(text).sentiment.polarity
    vd = _analyzer.polarity_scores(text)["compound"]
    score = 0.5 * tb + 0.5 * vd
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


# ─── 2) DATA LOADING & SPLIT ───────────────────────────────────────────────────

def load_and_split(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    df = pd.read_csv(csv_path)
    df = df[df["text_cleaned"].notna()]

    if "sentiment" not in df.columns:
        df["sentiment"] = df["text_cleaned"].apply(weak_sentiment)

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["sentiment"].map(label_map)

    X = df["text_cleaned"].tolist()
    y = df["label"].values

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


# ─── 3) TRANSFORMER TOKENIZATION ────────────────────────────────────────────────

def encode_texts(tokenizer, texts, max_len=200):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )


# ─── 4) MODEL BUILDING & TRAINING ──────────────────────────────────────────────

def train_and_evaluate(
    input_csv: str,
    model_out: str,
    tokenizer_out: str,
    report_out: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_len: int = 200,
    batch_size: int = 32,
    epochs: int = 4,
    lr: float = 3e-5
):
    # 1. Load & split
    X_train, X_val, y_train, y_val = load_and_split(
        input_csv, test_size, random_state
    )

    # 2. Tokenize
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )
    enc_train = encode_texts(tokenizer, X_train, max_len)
    enc_val   = encode_texts(tokenizer, X_val,   max_len)

    # 3. Build model
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )
    optimizer = Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=model.compute_loss,
        metrics=["accuracy"]
    )

    # 4. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_out,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Train
    history = model.fit(
        x=enc_train.data,
        y=y_train,
        validation_data=(enc_val.data, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )

    # 6. Evaluate
    preds = model.predict(enc_val.data).logits.argmax(axis=1)
    report = classification_report(
        y_val, preds,
        target_names=["negative", "neutral", "positive"],
        output_dict=True
    )
    cm = confusion_matrix(y_val, preds)

    # 7. Save artifacts
    model.save_pretrained(os.path.dirname(model_out))
    tokenizer.save_pretrained(os.path.dirname(tokenizer_out))
    pd.DataFrame(report).transpose().to_csv(report_out)

    return history, cm, report


# ─── 5) ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv",        required=True)
    p.add_argument("--model_out",  required=True)
    p.add_argument("--tokenizer_out", required=True)
    p.add_argument("--report_out", required=True)
    args = p.parse_args()

    hist, cm, rpt = train_and_evaluate(
        input_csv=args.csv,
        model_out=args.model_out,
        tokenizer_out=args.tokenizer_out,
        report_out=args.report_out
    )

    print("Final classification report:")
    print(pd.DataFrame(rpt).transpose())
