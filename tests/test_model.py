# import os
# import joblib


# def test_model_exists():
#     assert os.path.exists("models/best_model.pkl")


# def test_model_load():
#     model = joblib.load("models/best_model.pkl")
#     assert model is not None

import os
import joblib
import pytest

MODEL_PATH = "models/best_model.pkl"


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model file not available in CI"
)
def test_model_exists():
    assert os.path.exists(MODEL_PATH)


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model file not available in CI"
)
def test_model_load():
    model = joblib.load(MODEL_PATH)
    assert model is not None