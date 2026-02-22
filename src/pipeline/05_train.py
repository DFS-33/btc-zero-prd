#!/usr/bin/env python3
"""05_train.py -- Train multiple classifiers and select the best by Recall.

Trains RandomForest, XGBoost, LightGBM, and LogisticRegression on the
preprocessed training set. Evaluates each on the validation set using Recall
as the primary metric. Serializes the model with the highest validation Recall
to model/best_model.pkl.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    """Load pipeline configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_models(config: dict) -> dict:
    """Instantiate all classifiers from config hyperparameters.

    Args:
        config: Pipeline configuration with training.models section.

    Returns:
        Dictionary mapping model name to unfitted classifier instance.
    """
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    model_configs = config["training"]["models"]

    models: dict = {
        "RandomForest": RandomForestClassifier(
            n_estimators=model_configs["RandomForest"]["n_estimators"],
            max_depth=model_configs["RandomForest"]["max_depth"],
            class_weight=model_configs["RandomForest"]["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=model_configs["XGBoost"]["n_estimators"],
            max_depth=model_configs["XGBoost"]["max_depth"],
            learning_rate=model_configs["XGBoost"]["learning_rate"],
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=model_configs["LightGBM"]["n_estimators"],
            max_depth=model_configs["LightGBM"]["max_depth"],
            learning_rate=model_configs["LightGBM"]["learning_rate"],
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=model_configs["LogisticRegression"]["max_iter"],
            class_weight=model_configs["LogisticRegression"]["class_weight"],
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
    }

    logger.info("Built %d model instances: %s", len(models), list(models.keys()))
    return models


def train_model(
    name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Fit a single classifier on the training data.

    Args:
        name: Model name for logging.
        model: Unfitted sklearn-compatible classifier.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted classifier.
    """
    logger.info("Training %s ...", name)
    model.fit(X_train, y_train)
    return model


def evaluate_recall(model: object, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Compute Recall on the positive class for a fitted model.

    Args:
        model: Fitted classifier with a predict() method.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Recall score on the positive class (label=1).
    """
    y_pred = model.predict(X_val)
    return float(recall_score(y_val, y_pred, pos_label=1, zero_division=0))


def select_best(results: dict) -> tuple[str, object, float]:
    """Select the model with the highest validation Recall.

    Args:
        results: Dictionary mapping model name to {'model': ..., 'val_recall': float, ...}.

    Returns:
        Tuple of (best_name, best_model, best_recall).
    """
    best_name = max(results, key=lambda k: results[k]["val_recall"])
    best_entry = results[best_name]
    return best_name, best_entry["model"], best_entry["val_recall"]


def save_model(model: object, path: Path) -> None:
    """Serialize the model to disk using joblib.

    Args:
        model: Fitted classifier to serialize.
        path: Output file path (.pkl).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def save_results(results: dict, best_name: str, best_recall: float, path: Path) -> None:
    """Write training comparison summary to a JSON file.

    Args:
        results: Per-model training results (without model objects).
        best_name: Name of the selected best model.
        best_recall: Validation Recall of the best model.
        path: Output JSON file path.
    """
    summary = {
        "best_model": best_name,
        "best_val_recall": round(best_recall, 4),
        "model_comparison": {
            name: {k: v for k, v in entry.items() if k != "model"}
            for name, entry in results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Training summary saved to %s", path)


def main() -> None:
    """Load splits, train all models, select best by Recall, save model and summary."""
    config = load_config()

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    X_train = pd.read_csv(processed_dir / "X_train.csv", encoding="utf-8")
    y_train = pd.read_csv(processed_dir / "y_train.csv", encoding="utf-8").squeeze()
    X_val = pd.read_csv(processed_dir / "X_val.csv", encoding="utf-8")
    y_val = pd.read_csv(processed_dir / "y_val.csv", encoding="utf-8").squeeze()

    logger.info(
        "Data loaded -- train: %d rows | val: %d rows",
        len(X_train), len(X_val),
    )

    models = build_models(config)
    cv_folds = config["training"]["cv_folds"]
    results: dict = {}

    for name, model in models.items():
        fitted_model = train_model(name, model, X_train, y_train)
        val_recall = evaluate_recall(fitted_model, X_val, y_val)

        cv_scores = cross_val_score(
            fitted_model, X_train, y_train,
            cv=cv_folds, scoring="recall", n_jobs=-1,
        )

        report = classification_report(
            y_val, fitted_model.predict(X_val), output_dict=True, zero_division=0,
        )

        results[name] = {
            "model": fitted_model,
            "val_recall": round(val_recall, 4),
            "cv_recall_mean": round(float(cv_scores.mean()), 4),
            "cv_recall_std": round(float(cv_scores.std()), 4),
            "val_precision": round(float(report.get("1", {}).get("precision", 0.0)), 4),
            "val_f1": round(float(report.get("1", {}).get("f1-score", 0.0)), 4),
        }

        logger.info(
            "%s -- Val Recall: %.4f | CV Recall: %.4f (+/- %.4f)",
            name, val_recall, cv_scores.mean(), cv_scores.std(),
        )

    best_name, best_model, best_recall = select_best(results)
    logger.info("Best model: %s with Val Recall=%.4f", best_name, best_recall)

    model_dir = PROJECT_ROOT / config["output"]["model_dir"]
    model_path = model_dir / config["output"]["model_filename"]
    save_model(best_model, model_path)

    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    summary_path = metrics_dir / config["output"]["training_summary_filename"]
    save_results(results, best_name, best_recall, summary_path)


if __name__ == "__main__":
    main()
