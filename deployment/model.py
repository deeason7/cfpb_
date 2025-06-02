from tensorflow.keras.models import load_model as keras_load
import os

from dotenv import load_dotenv
load_dotenv()

class ModelLoader:
    """
    A helper class to hold a trained Keras BiLSTM model from disk.
    """

    def __init__(self, model_path: str = None):
        """
        Args: 
             model_path (str): If not provided, will try to read MODEL_PATH from env
        """
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH")

            if model_path is None:
                raise ValueError("MODEL_PATH environment variable is not set.")
        self.model_path = model_path
        self.model = None

    def validate_path(self):
        """
        Ensures that 'self.model_path' exists and is readable.

        Raises:
            FileNotFoundError: If the given model_path does not exist
        """

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not fount at: {self.model_path}")

    def load(self):
        """
        Load the Keras model into memory

        Returns:
            keras.Model: The loaded model instance, ready for inference

        Raises:
            FileNotFoundError: If 'model_path' is invalid.
            ValueError: If loading fails (e.g., corrupted file).
        """
        self.validate_path()
        # Use Keras's load_model under the hood

        try:
            self.model = keras_load(self.model_path, compile=False)
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")
        return self.model

# Sanity check
if __name__ == "__main__":
    try:
        loader = ModelLoader()
        model = loader.load()
        print(f"Model loaded successfully: {model}")
    except Exception as err:
        print(f"{err}")