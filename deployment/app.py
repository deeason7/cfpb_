# deployment/app.py
"""
CLI Application to tie together ModelLoader, Predictor, and Storage for an end-to-end demo.
"""

import os
import datetime
from dotenv import load_dotenv

from deployment.model import ModelLoader
from deployment.predictor import Predictor
from deployment.storage import Storage

def main():
    # Load environment variables from .env
    load_dotenv()


    # 1. Fetch paths from environment or defaults
    MODEL_PATH = os.environ.get("MODEL_PATH")
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")
    LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH")
    DB_PATH = os.environ.get("DB_PATH")

    # 2. Verify artifact existence
    missing = []
    for path in (MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH):
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("Missing required file(s):")
        for p in missing:
            print(f"   - {p}")
        print("Please update paths or environment variables and try again.")
        return

    # 3. Initialize ModelLoader and load model
    try:
        loader = ModelLoader(MODEL_PATH)
        model = loader.load()
        print("Model loaded successfully.")
    except Exception as err:
        print(f"Error loading model: {err}")
        return

    # 4. Initialize Predictor
    try:
        predictor = Predictor(
            model=model,
            tokenizer_path=TOKENIZER_PATH,
            label_encoder_path=LABEL_ENCODER_PATH,
            max_len=250,
            struct_dim=5
        )
        print(" Predictor initialized successfully.")
    except Exception as err:
        print(f"  Error initializing Predictor: {err}")
        return

    # 5. Initialize Storage (SQLite)
    try:
        storage = Storage(db_path= DB_PATH)
        print(f"SQLite database ready at: {DB_PATH}")
    except Exception as err:
        print(f"Error initializing Storage: {err}")
        return

    #6. Interactive loop
    print("\n Deployment Demo CLI")
    print("Enter complaint narrative (or type 'exit' to quit): \n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break

        if not user_input:
            print("Please enter some text or 'exit' to quit.")
            continue

        # Run prediction
        try:
            result = predictor.predict(user_input)
        except Exception as err:
            print(f"Prediction error: {err}")
            continue

        # Extract Results
        label = result["label"]
        confidence = result["confidence"]
        struct_feats = result["structured_features"]

        #Display results
        print(f"Prediction: ")
        print(f"    Label:        {label}")
        print(f"    Confidence:   {confidence:.2f}")
        print("     Structured features used:")
        for feat_name, feat_val in struct_feats.items():
            print(f"           -{feat_name}: {feat_val}")
        print()

        #Log to SQLite
        timestamp = datetime.datetime.now().isoformat()
        try:
            storage.log(timestamp, user_input, label, confidence, [str(v) for v in struct_feats.values()])
            print(f"Logged prediction at {timestamp}\n")

        except Exception as err:
            print(f"Failed to log prediction: {err}\n")

if __name__ == "__main__":
    main()
