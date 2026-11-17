"""
FastAPI application for customer churn prediction.

Provides prediction, health-check, readiness, and model information
endpoints for Kubernetes-based machine learning model serving.
"""

from contextlib import asynccontextmanager
from pathlib import Path