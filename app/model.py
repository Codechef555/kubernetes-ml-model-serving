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
    
    def load_model(self) -> None:
        """Load the serialized machine learning model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Run 'python -m ml.train' to train the model first."
            )

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the trained model: {exc}"
            ) from exc
    
    def is_loaded(self) -> bool:
        """Check whether the model is loaded."""
        return self.model is not None