from src.data_loader import load_data
from src.model import train_model, evaluate_model

def test_model_training():
    df = load_data("")
    model, X_test, y_test = train_model(df)

    assert model is not None

def test_model_evaluation():
    df = load_data("")
    model, X_test, y_test = train_model(df)

    mse = evaluate_model(model, X_test, y_test)

    assert mse >= 0 