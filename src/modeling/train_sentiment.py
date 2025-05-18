#Import required libraries
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json

# Scikit-learn tools for preprocessing and evaluation
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# TensorFlow and Keras modules for model building
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam

# Ensure parent directory is in the system path
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)

# Function to prepare text and structured data
def prepare_data(path="../data/processed/consumer_complaints_final.csv", vocab_size=25000, max_len=250):
    """
    Load and preprocess data for sentiment classification

    Performs:
    - Loads cleaned complaint data from CSV.
    - Encodes sentiment labels into integers using LabelEncoder.
    - Splits the features into text data and structured numeric features.
    - Tokenizes and pads the text data to a fixed maximum length.
    - Splits the dataset into training, validation, and test subsets using stratified sampling.

    Args:
        path(str): File path to the prerprocessed CSV data.
        vocab_size(int): Maximum number of words to keep in the tokenizer vocabulary.
        max_len(int): Maximum sequence length for padding/truncation.

    Returns:
        Tuple:
            X_seq_train (ndarray): Padded training sequences.
            X_struct_train (ndarray): Structured training features.
            y_train (ndarray): Encoded training labels.
            X_seq_val(ndarray): Padded validation sequences.
            X_struct_val(ndarray): Structured validation features.
            y_val (ndarray): Encoded validation labels.
            X_seq_test(ndarray): Padded test sequences.
            X_struct_test(ndarray): Structured test features.
            y_test(ndarray): Encoded test labels.
            tokenizer(Tokenizer): Fitted Keras tokenizer object.
            le(LabelEncoder): Fitted label encoder object for inverse mapping.
    """
    df = pd.read_csv(path)
    #Remove rows with null cleaned text
    df = df[df['text_cleaned'].notna()].copy()

    # Encode Sentiment labels into integers
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

    # Split into text and structured input features
    X_text = df['text_cleaned']
    X_struct = df[['text_length', 'timely_response_binary', 'product_dispute_rate',
                   'company_dispute_rate', 'keyword_flag']].values.astype(np.float32)
    y = df['sentiment_encoded']

    # Tokenize and pad text data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_text)
    X_seq = pad_sequences(tokenizer.texts_to_sequences(X_text), maxlen=max_len)

    # Create train, validation, and test sets (70-15-15 split)
    X_seq_train, X_seq_temp, X_struct_train, X_struct_temp, y_train, y_temp = train_test_split(
        X_seq, X_struct, y, test_size=0.3, stratify=y, random_state=42
    )
    X_seq_val, X_seq_test, X_struct_val, X_struct_test, y_val, y_test = train_test_split(
        X_seq_temp, X_struct_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return (X_seq_train, X_struct_train, y_train,
            X_seq_val, X_struct_val, y_val,
            X_seq_test, X_struct_test, y_test,
            tokenizer, le)

# Function to build a hybrid LSTM + structured model 
def build_model(vocab_size=25000, max_len=250):
    #Input layers for text and structured data
    text_input = Input(shape=(max_len,), name="text_input")
    struct_input = Input(shape=(5,), name="struct_input")
    
    # Text branch with embedding and BiLSTM
    x = Embedding(input_dim=vocab_size, output_dim=128)(text_input)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)

    #Merged with structured inputs
    merged = Concatenate()([x, struct_input])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    output = Dense(3, activation='softmax')(merged) # 3 sentiment classes

    # Compile the model
    model = Model(inputs=[text_input, struct_input], outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    return model

# Full training, evaluation, and artifact saving pipeline
def train_and_evaluate_sentiment():
    # Load and preprocess data
    (X_seq_train, X_struct_train, y_train,
     X_seq_val, X_struct_val, y_val,
     X_seq_test, X_struct_test, y_test,
     tokenizer, le) = prepare_data()

    # Compute class weights to handle class imbalance
    class_weights = dict(
        zip(np.unique(y_train),
            compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train))
    )

    # Build the model
    model = build_model()
    
    # Callbacks for better training control
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    # Model training
    history = model.fit(
        {"text_input": X_seq_train, "struct_input": X_struct_train}, y_train,
        validation_data=({"text_input": X_seq_val, "struct_input": X_struct_val}, y_val),
        epochs=30,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop, lr_schedule],
        verbose=2
    )

    # Final test evaluation
    y_pred_prob = model.predict({"text_input": X_seq_test, "struct_input": X_struct_test})
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Save model and preprocessing artifacts
    model.save("../models/sentiment_model.keras")

    with open("../outputs/tokenizer_sentiment.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("../outputs/label_encoder_sentiment.pkl", "wb") as f:
        pickle.dump(le, f)

    # Save evaluation report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    with open("../outputs/sentiment_accuracy.txt", "w") as f:
        json.dump(report, f, indent=2)

    # Save predictions for inspection
    predictions_df = pd.DataFrame({
        "true_sentiment": le.inverse_transform(y_test),
        "predicted_sentiment": le.inverse_transform(y_pred)
    })
    predictions_df.to_csv("../outputs/predictions_sentiment.csv", index=False)

    print("Sentiment model trained and evaluated on test data.")

# Execute the training pipeline when run directly
if __name__ == '__main__':
    train_and_evaluate_sentiment()
