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
    
    def predict(self, customer_data: dict) -> dict:
        """
        Predict whether a customer is likely to churn.

        Args:
            customer_data: Dictionary containing customer features.

        Returns:
            Dictionary containing prediction and probability.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Model is not loaded. Call load_model() first."
            )

        if not isinstance(customer_data, dict) or not customer_data:
            raise ValueError("Customer data must be a non-empty dictionary.")

        try:
            input_data = pd.DataFrame([customer_data])

            prediction = int(self.model.predict(input_data)[0])

            probabilities = self.model.predict_proba(input_data)[0]

            classes = list(self.model.classes_)
            churn_class_index = classes.index(1)

            churn_probability = float(
                probabilities[churn_class_index]
            )

            prediction_label = (
                "Churn" if prediction == 1 else "No Churn"
            )

            return {
                "prediction": prediction_label,
                "churn_probability": round(churn_probability, 4),
                "confidence": round(
                    float(max(probabilities)), 4
                ),
            }

        except (ValueError, TypeError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Customer churn prediction failed: {exc}"
            ) from exc

# Reusable model instance for the FastAPI application.
churn_model = ChurnPredictionModel()