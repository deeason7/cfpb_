import pandas as pd
import numpy as np
import pickle
import os
import sys
import json

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from textblob import TextBlob

# Add src/ path to sys
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)


def weak_sentiment(text: str) -> str:
    """
    Rule‐based sentiment labeling using TextBlob polarity:
    - polarity > 0.1 → positive
    - polarity < -0.1 → negative
    - else → neutral
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "neutral"
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"


def prepare_data(path="../data/processed/enhanced_consumer_complaints.csv", vocab_size=15000, max_len=250):
    """
    Load and preprocess text data for sentiment classification.
    Returns encoded sequences and labels, tokenizer, and label encoder.
    """
    df = pd.read_csv(path)
    df = df[df['text_cleaned'].notna()].copy()

    # Weak sentiment labeling
    df['sentiment'] = df['text_cleaned'].apply(weak_sentiment)

    # Encode labels
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

    X = df['text_cleaned']
    y = df['sentiment_encoded']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Tokenize
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len)

    return X_train_seq, X_val_seq, y_train, y_val, tokenizer, le


def build_model(vocab_size=15000, max_len=250):
    """
    BiLSTM model with dropout for sentiment classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 sentiment classes
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate_sentiment():
    """
    Train the sentiment classification model, save outputs.
    """
    X_train_seq, X_val_seq, y_train, y_val, tokenizer, le = prepare_data()

    # Compute class weights
    class_weights = dict(
        zip(np.unique(y_train),
            compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train))
    )

    # Build model
    model = build_model()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    # Train
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=20,
        batch_size=128,
        class_weight=class_weights,
        callbacks=[early_stop, lr_schedule],
        verbose=2
    )

    # Predict
    y_pred_prob = model.predict(X_val_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Save model
    model.save("../models/sentiment_model.keras")

    # Save tokenizer
    with open("../outputs/tokenizer_sentiment.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Save label encoder
    with open("../outputs/label_encoder_sentiment.pkl", "wb") as f:
        pickle.dump(le, f)

    # Save evaluation report
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    with open("../outputs/sentiment_accuracy.txt", "w") as f:
        json.dump(report, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame({
        "true_sentiment": le.inverse_transform(y_val),
        "predicted_sentiment": le.inverse_transform(y_pred)
    })
    predictions_df.to_csv("../outputs/predictions_sentiment.csv", index=False)

    print(" Sentiment model trained, evaluated, and outputs saved.")
