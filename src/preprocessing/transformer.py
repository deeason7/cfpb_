# src/preprocessing/transformer.py

import pandas as pd

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform structured features, prune irrelevant columns, and add derived features.
    """
    # Encode binary columns
    df["consumer_disputed_binary"] = df["consumer_disputed?"].map({"Yes": 1, "No": 0})
    df["timely_response_binary"] = df["timely_response"].map({"Yes": 1, "No": 0})

    # Encode categorical company response
    df["company_response_encoded"] = df["company_response_to_consumer"].astype("category").cat.codes

    # Add text length feature
    df["text_length"] = df["text_cleaned"].apply(lambda x: len(x.split()))

    # Replace missing state info
    df["state"] = df["state"].fillna("Unknown")

    # Drop columns not used in modeling or analysis
    cols_to_drop = [
        "consumer_complaint_narrative",  # raw text
        "complaint_id",                  # ID not needed for ML
        "date_received",                # datetime not used here
        "date_sent_to_company",
        "submitted_via",
        "zipcode",                      # removed from visualization
        "sub_issue",     # too sparse or redundant
        "consumer_disputed?",           # replaced with binary
        "company_response_to_consumer"  # replaced with encoded
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors="ignore")

    return df
