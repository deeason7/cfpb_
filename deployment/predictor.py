"""
Predictor module for Deployment:
Loads tokenizer and label encoder artifacts, preprocess user inputs, and runs inference using the trained hybrid BiLSTM + structured- input model.
"""

import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv
from typing import List, Any

class Predictor:
    """
    Encapsulates loading artifacts (tokenizer, label encoder), preprocessing of text and dummy structured inputs (for CLI demo),
    and model inference.
    """

    def __init__(self,
                 model,
                 tokenizer_path: str = None,
                 label_encoder_path: str= None,
                 max_len: int = 250,
                 struct_dim: int = 5):
        """
        Args:
            model: A loaded Keras model expecting two inputs:
                    - text_input of shape (None, max_len)
                    - struct_input of shape (None, struct_dim)
            tokenizer_path (str): Path to the pickled Keras Tokenizer
            label_encoder_path(str): Path to the pickled LabelEncoder
            max_len(int): Sequence length used during training (default: 250)
            struct_dim(int): Dimension of structured feature vector (default: 5)
        """
        self.model = model
        self.max_len = max_len
        self.struct_dim = struct_dim

        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = os.environ.get("TOKENIZER_PATH")

            if tokenizer_path is None:
                raise ValueError("TOKENIZER_PATH environment variable is not set.")

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        #Load label encoder
        if label_encoder_path is None:
            label_encoder_path = os.environ.get("LABEL_ENCODER_PATH")

            if label_encoder_path is None:
                raise ValueError("LABEL_ENCODER_PATH environment variable is not set")

        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def preprocess_text(self, text: str) -> np.ndarray:
        """
        Tokenizes and pads a single text string to shape (1, max_len)

        Args:
            text(str): Raw complaint narrative.

        Returns:
            np.ndarray: Padded token IDs with shape (1, max_len)
        """
        normalized = text.lower()
        seq = self.tokenizer.texts_to_sequences([normalized])
        padded = pad_sequences(seq, maxlen=self.max_len, padding= 'post', truncating='post')
        return padded # shape: (1, max_len)

    def preprocess_struct(self, text:str) -> np.ndarray:
        """
        For ClI demo, I created a dummmy structured-input vector. In a production pipeline, it is necessary to compute features:
        - text_length
        - timely_response_binary
        - product_dispute_rate
        - company_dispute_rate
        - keyword_flag

        Here, approximation is:
        text_length = len(text.split())
        timely_response_binary = 0
        product_dispute_rate = 0.0
        company_dispute_rate = 0.0
        keyword_flag = 1 if any(f in text.lower() for f in ["fraud","lawsuit"]) else 0

        Returns:
            np.ndarray: Array of shape(1, struct_dim) with floats.
        """
        tokens = text.split()
        text_length = len(tokens)
        timely_response_binary = 0 # Placeholder for CLI demo
        product_dispute_rate = 0.0 # Placeholder for CLI demo
        company_dispute_rate = 0.0 # Placeholder for CLI demo
        keyword_flag = 1 if any(kw in text.lower() for kw in ["fraud", "lawsuit", "chargeback"]) else 0

        struct_vec = np.array(
            [[
                float(text_length),
                float(timely_response_binary),
                float(product_dispute_rate),
                float(company_dispute_rate),
                float(keyword_flag)
            ]]
        )
        return struct_vec # shape: (1,5)

    def predict(self, text:str) -> dict:
        """
        Runs the model on a raw text input and returns:
        - label(str): e.g., 'Neutral'
        - confidence(float): Probability of the predicted class
        -structured_features(dict): The dummy structured features used
        - Note: Keyword extraction (attention) is ommited in this CLI demo.

        Args:
            text(str): Complaint narrative

        Returns:
            dict:{
                'label': str,
                'confidence':float,
                'structured_features': {
                    'text_length': float,
                    'timely_response_binary': int,
                    'product_dispute_rate': float,
                    'company_dispute_rate': float,
                    'keyword_flag': int
                }

                    }
        """
        # Preprocess text and structured inputs
        padded_seq = self.preprocess_text(text)  # shape: (1, max_len)
        struct_vec = self.preprocess_struct(text) # shape: (1, struct_dim)

        #Predict class probabilities
        probs = self.model.predict([padded_seq, struct_vec], verbose = 0) #e.g., [[0,.1, 0.7, 0.2]]
        probs = probs.flatten()

        #Determine predicted index and confidence
        pred_index = int(np.argmax(probs))
        confidence = float(probs[pred_index])

        #Convert index->label string via LabelEncoder
        label = self.label_encoder.inverse_transform([pred_index])[0]

        #Returns results
        return {
            'label': label,
            'confidence': confidence,
            'structured_features': {
                'text_length': float(len(text.split())),
                'timely_response_binary': 0,
                'product_dispute_rate': 0.0,
                'company_dispute_rate': 0.0,
                'keyword_flag': 1 if any(kw in text.lower() for kw in ["fraud", "lawsuit", "chargeback"]) else 0
            }
        }


#-----------------------------Sanity Check-----------------------------
if __name__ == "__main__":
    """
    Sanity check for Predictor:
    - Loads a saved model, tokenizer, and label encoder from disk or environment.
    - Runs a sample prediction on a dummy complaint text.
    - Prints out the label, confidence, and structured features.
    """

    import tensorflow as tf
    from deployment.model import ModelLoader

    # Load environment variables from a .env file if present
    load_dotenv()

    # 1. Define or fetch path to artifacts
    MODEL_PATH = os.environ.get("MODEL_PATH")
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")
    LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH")

    # 2. Verify that files exist
    missing = []
    for p in (MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH):
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        print("Sanity check failed- could not find these files:")
        for p in missing:
            print("    -", p)
        print("\n Please update the paths or environment variables and retry.")
        exit(1)

    # 3. Load the Keras model
    try:
        loader = ModelLoader(MODEL_PATH)
        model = loader.load()
        print("Model loaded successfully")

    except Exception as err:
        print(f"Failed to load model: {err}")
        exit(1)

    # 4. Instantiate Predictor
    try:
        predictor = Predictor(
            model = model,
            tokenizer_path=TOKENIZER_PATH,
            label_encoder_path = LABEL_ENCODER_PATH,
            max_len=250,
            struct_dim=5
        )
        print("Predictor initialized successfully.")

    except Exception as err:
        print(f"Failed to initialize Predictor: {err}")
        exit(1)

    # Run a sample prediction
    SAMPLE_TEXT =(
        "I was charged an unexpected fee for fraud protection services."
        "I have tried contacting customer service multiple times with no response!"
    )

    print(f"\n⊳ Running prediction on sample text: \n\n {SAMPLE_TEXT}")
    try:
        result = predictor.predict(SAMPLE_TEXT)
        print("Prediction result: ")
        print(f"   Label: {result['label']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print("    Structured features:")
        for k, v in result["structured_features"].items():
            print(f"     - {k}: {v}")

    except Exception as err:
        print(f" Prediction failed: {err}")
        exit(1)

    print("Sanity check passed. Predictor is functioning as expected>")


