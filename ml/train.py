import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "ml"


DATASET_PATH = DATA_DIR / "customer_churn.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the customer churn dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Place customer_churn.csv inside the data/ directory."
        )
    
    dataframe = pd.read_csv(dataset_path)

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    return dataframe

def preprocess_dataset(dataframe: pd.DataFrame):
    """Separate features and target and prepare preprocessing pipelines."""
    dataframe = dataframe.copy()

    # Remove customer identifiers because they do not predict churn.
    for identifier in ("customerID", "CustomerID", "customer_id"):
        if identifier in dataframe.columns:
            dataframe.drop(columns=[identifier], inplace=True)

    dataframe[TARGET_COLUMN] = (
        dataframe[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    dataframe.dropna(subset=[TARGET_COLUMN], inplace=True)

    if dataframe[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The target column must contain both churn classes.")

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN].astype(int)

    if X.empty or X.shape[1] == 0:
        raise ValueError("The dataset contains no usable features.")
    
    numerical_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    transformers = []
    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numerical", numerical_pipeline, numerical_features)
        )
    
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )
