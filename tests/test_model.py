"""Tests for model training and evaluation."""

from pathlib import Path
from src.data_loader import load_data
from src.model import train_model, evaluate_model


def get_data_path():
    """Return correct path to dataset."""
    return Path(__file__).resolve().parents[1] / "data" / "student_scores_updated.csv"


def test_model_training():
    """Test model training."""
    df = load_data(get_data_path())
    model, X_test, y_test = train_model(df)

    assert model is not None


def test_model_evaluation():
    """Test model evaluation."""
    df = load_data(get_data_path())
    model, X_test, y_test = train_model(df)

    mse = evaluate_model(model, X_test, y_test)
    assert mse >= 0
    