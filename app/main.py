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
