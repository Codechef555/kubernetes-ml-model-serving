"""
Load the trained customer churn model and provide prediction functionality.
"""
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "model.pkl"

class ChurnPredictionModel:
    """Reusable interface for customer churn predictions."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
