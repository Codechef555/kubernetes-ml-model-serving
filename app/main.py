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

# Initialize FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=(
        "Machine learning API for predicting customer churn "
        "using a trained Random Forest classification model."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# Expose Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())

@app.get("/", response_model=HealthResponse, tags=["General"])
async def root():
    """Return the API welcome message."""
    return HealthResponse(
        status="success",
        message="Customer Churn Prediction API is running.",
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Liveness endpoint.

    Indicates whether the API process is running.
    """
    return HealthResponse(
        status="healthy",
        message="Application is running.",
    )

@app.get("/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness endpoint.

    Indicates whether the trained ML model is available
    to serve prediction requests.
    """
    if not churn_model.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Application is not ready. Model is not loaded.",
        )

    return HealthResponse(
        status="ready",
        message="Application and ML model are ready.",
    )

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
async def model_information():
    """Return information about the deployed ML model."""
    if not churn_model.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded.",
        )

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_status="loaded",
        target_column="Churn",
        supported_predictions=["Churn", "No Churn"],
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict whether a customer is likely to churn.

    Accepts validated customer information and returns
    the predicted churn class and probability.
    """
    if not churn_model.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="ML model is not available.",
        )

    PREDICTION_REQUESTS.inc()

    try:
        with PREDICTION_LATENCY.time():
            result = churn_model.predict(
                customer.model_dump()
            )

        prediction = result["prediction"]

        return PredictionResponse(
            prediction=prediction,
            churn_probability=result["churn_probability"],
            confidence=result["confidence"],
            message=(
                "Customer is likely to churn."
                if prediction == "Churn"
                else "Customer is unlikely to churn."
            ),
        )

    except Exception as exc:
        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

@app.get("/metrics-info", tags=["Monitoring"])
async def metrics_information():
    """Provide information about the available monitoring endpoint."""
    return {
        "metrics_endpoint": "/metrics",
        "available_metrics": [
            "prediction_requests_total",
            "prediction_errors_total",
            "prediction_latency_seconds",
        ],
        "monitoring": "Prometheus",
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )