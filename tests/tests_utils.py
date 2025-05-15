#tests/test_utils.py
from tensorflow.keras.layers import InputLayer
import pickle

import pytest
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

from cfpb_depolyment.utils import clean_text
from cfpb_depolyment import utils


def test_clean_text_removes_html_and_non_alphanumerics():
    raw = "<p>Hello <b> World</b>!!! Visit https://example.com</p>"
    cleaned = clean_text(raw)
    # should be lowercase, tags stripped, punctuation/URLs removed
    assert "hello world" in cleaned
    assert "<" not in cleaned and "https" not in cleaned

def test_clean_text_normalizes_whitespaces():
    raw = "      Multiple      spaces \n and \t\t   tabs   "
    cleaned = clean_text(raw)
    # multiple spaces/newlines/tabs -> single spaces, stripped ends
    assert cleaned == "multiple spaces and tabs"

def test_load_model_and_artifacts_succeeds(tmp_path, monkeypatch):
    # Create dummy model file
    model_path = tmp_path / "sentiment_model.h5"
    #empty model is enough to save
    model = Sequential([InputLayer(input_shape=(10,))])
    model.save(str(model_path), include_optimizer=False)

    #Create dummy tokenizer and encoder pickles
    tok_path = tmp_path / "tokenizer.pkl"
    enc_path = tmp_path / "label_encoder.pkl"
    with open(tok_path, "wb") as f:
        pickle.dump(Tokenizer(), f)
    with open(enc_path, "wb") as f:
        pickle.dump(LabelEncoder(), f)

    #Monkey patch the path in utils
    monkeypatch.setattr(utils, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(utils, "TOKENIZER_PATH", str(tok_path))
    monkeypatch.setattr(utils, "ENCODER_PATH", str(enc_path))

    loaded_model, loaded_tokenizer, loaded_encoder = utils.load_model_and_artifacts()

    #Basic sanity checks
    assert hasattr(loaded_model, "predict")
    assert isinstance(loaded_tokenizer, Tokenizer)
    assert isinstance(loaded_encoder, LabelEncoder)

def test_load_model_and_artifacts_missing_files(monkeypatch):
    #If any file is missing we expect a FileNotFoundError
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(utils,"MODEL_PATH", "nonexistent_model.h5")
    monkeypatch.setattr(utils, "TOKENIZER_PATH",   "nonexistent_tokenizer.pkl")
    monkeypatch.setattr(utils,"ENCODER_PATH", "nonexistent_encoder.pkl")
    with pytest.raises(FileNotFoundError):
        utils.load_model_and_artifacts()
    monkeypatch.undo()