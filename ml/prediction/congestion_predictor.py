"""Scikit-Learn congestion prediction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.prediction.dataset_builder import DEFAULT_OUTPUT_PATH, build_training_dataset


MODEL_OUTPUT_PATH = Path("data/models/congestion_predictor.pkl")
FEATURE_COLUMNS = [
    "total_vehicles",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "density",
    "emergency_present",
]
TARGET_COLUMN = "congestion"
NUMERIC_FEATURES = ["total_vehicles", "car_count", "motorcycle_count", "bus_count", "truck_count"]
CATEGORICAL_FEATURES = ["density", "emergency_present"]


def _load_training_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    """Load and validate a generated training dataset."""

    path = Path(dataset_path)
    if not path.exists():
        build_training_dataset(output_path=path)

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Training dataset is missing columns: {missing_columns}")
    if dataframe.empty:
        raise ValueError("Training dataset is empty")

    dataframe = dataframe.copy()
    dataframe["emergency_present"] = dataframe["emergency_present"].astype(str)
    return dataframe


def _build_pipeline() -> Pipeline:
    """Create the Scikit-Learn training pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate evaluation metrics for the trained model."""

    predictions = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
    }


def train_model(
    dataset_path: str | Path = DEFAULT_OUTPUT_PATH,
    model_output_path: str | Path = MODEL_OUTPUT_PATH,
) -> dict[str, Any]:
    """Train and persist the congestion prediction model."""

    dataframe = _load_training_dataframe(dataset_path)
    x = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    stratify = y if y.value_counts().min() >= 2 and len(y.unique()) > 1 else None
    if len(dataframe) >= 6 and len(y.unique()) > 1:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
    else:
        x_train, x_test, y_train, y_test = x, x, y, y

    model = _build_pipeline()
    model.fit(x_train, y_train)
    metrics = _evaluate_model(model, x_test, y_test)

    output = Path(model_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)

    return {
        "model_path": str(output),
        "dataset_path": str(dataset_path),
        "training_rows": int(len(dataframe)),
        "classes": sorted(str(value) for value in y.unique()),
        "metrics": metrics,
    }


def load_model(model_path: str | Path = MODEL_OUTPUT_PATH) -> Pipeline:
    """Load a persisted congestion prediction model."""

    return joblib.load(model_path)


def _feature_frame(features: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row feature frame for prediction."""

    row = {column: features.get(column, 0) for column in FEATURE_COLUMNS}
    row["density"] = str(row.get("density", "LOW")).upper()
    row["emergency_present"] = str(bool(row.get("emergency_present", False)))
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_congestion(model: Pipeline, features: dict[str, Any]) -> str:
    """Predict congestion from one feature dictionary."""

    prediction = model.predict(_feature_frame(features))[0]
    return str(prediction)


def predict_from_model_file(
    model_path: str | Path = MODEL_OUTPUT_PATH,
    features: dict[str, Any] | None = None,
) -> str:
    """Load a saved model and run one prediction."""

    model = load_model(model_path)
    return predict_congestion(model, features or {})


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the model training and prediction CLI."""

    parser = argparse.ArgumentParser(description="Train or run congestion prediction.")
    parser.add_argument("--train", help="Path to generated training dataset CSV.")
    parser.add_argument("--model-output", default=str(MODEL_OUTPUT_PATH), help="Path to save trained model.")
    parser.add_argument("--model", default=str(MODEL_OUTPUT_PATH), help="Path to load a trained model.")
    parser.add_argument("--predict", action="store_true", help="Run prediction using CLI feature arguments.")
    parser.add_argument("--total-vehicles", type=int, default=0)
    parser.add_argument("--car-count", type=int, default=0)
    parser.add_argument("--motorcycle-count", type=int, default=0)
    parser.add_argument("--bus-count", type=int, default=0)
    parser.add_argument("--truck-count", type=int, default=0)
    parser.add_argument("--density", default="LOW")
    parser.add_argument("--emergency-present", action="store_true")
    return parser


def main() -> None:
    """Run training or prediction from the command line."""

    args = build_arg_parser().parse_args()

    if args.train:
        result = train_model(dataset_path=args.train, model_output_path=args.model_output)
        print(json.dumps(result, indent=2))
        return

    if args.predict:
        features = {
            "total_vehicles": args.total_vehicles,
            "car_count": args.car_count,
            "motorcycle_count": args.motorcycle_count,
            "bus_count": args.bus_count,
            "truck_count": args.truck_count,
            "density": args.density,
            "emergency_present": args.emergency_present,
        }
        prediction = predict_from_model_file(model_path=args.model, features=features)
        print(prediction)
        return

    print("Use --train DATASET.csv or --predict.")


if __name__ == "__main__":
    main()
