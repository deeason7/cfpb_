#src/modeling/train_dispute.py

import pandas as pd
import numpy as np
import pickle
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

#Prepare training data
def prepare_training_data(path="../data/processed/enhanced_consumer_complaints.csv"):
    """
    Load enhanced consumer complaints and prepare features for dispute modeling.
    """
    df = pd.read_csv(path)

    #Feature Engineering
    df['sentiment_timely_interaction'] = df['sentiment_encoded'] * df["timely_response_binary"]

    feature_cols = [
        "product_enc", "company_enc", "timely_response_binary", "text_length_norm", "sentiment_encoded", "product_disputed_rate",
        "sentiment_timely_interaction"
    ]

    X = df[feature_cols]
    y = df["consumer_disputed_binary"]

    #Standarize continuous columns
    scaler = StandardScaler()
    X[["text_length_norm", "product_dispute_rate"]] = scaler.fit_transform(X[["text_length_norm", "product_dispute_rate"]])

    #Save scaler
    with open(../outs)