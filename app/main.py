"""
FastAPI application for customer churn prediction.

Provides prediction, health-check, readiness, and model information
endpoints for Kubernetes-based machine learning model serving.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from app.model import churn_model
from app.schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# Application configuration
APP_NAME = "Customer Churn Prediction API"
APP_VERSION = "1.0.0"
MODEL_NAME = "RandomForestClassifier"

# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to generate customer churn predictions",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the application starts."""
    try:
        churn_model.load_model()
        print("Customer churn model loaded successfully.")
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        raise RuntimeError(
            "Application startup failed because the ML model "
            "could not be loaded."
        ) from exc

    yield

    print("Customer churn API shutting down.")