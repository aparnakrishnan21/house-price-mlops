import os
import joblib


def test_model_exists():
    assert os.path.exists("models/best_model.pkl")


def test_model_load():
    model = joblib.load("models/best_model.pkl")
    assert model is not None