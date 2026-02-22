#!/usr/bin/env python3
"""06_evaluate.py -- Final evaluation on the held-out test set.

Loads the serialized best model, evaluates it on the test set (never seen
during training or model selection), and writes metrics/report.json with
Recall, Precision, F1, ROC-AUC, confusion matrix, and feature importance.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    """Load pipeline configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(path: Path) -> object:
    """Deserialize a joblib-serialized model from disk.

    Args:
        path: Path to the .pkl file.

    Returns:
        Loaded classifier object.
    """
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


def evaluate(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute all evaluation metrics on the test set.

    Args:
        model: Fitted classifier with predict() and predict_proba() methods.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary with keys: recall, precision, f1, roc_auc, confusion_matrix,
        classification_report, and support per class.
    """
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = None
        logger.warning("ROC-AUC could not be computed")

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "recall": round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "test_set_size": len(y_test),
        "positive_class_count": int(y_test.sum()),
    }

    logger.info(
        "Test results -- Recall: %.4f | Precision: %.4f | F1: %.4f | ROC-AUC: %s",
        metrics["recall"], metrics["precision"], metrics["f1"],
        f"{roc_auc:.4f}" if roc_auc else "N/A",
    )
    return metrics


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap as a PNG image.

    Args:
        cm: Confusion matrix as a nested list.
        labels: Class labels for axis ticks.
        output_path: Path to save the PNG file.
    """
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        logger.info("Confusion matrix plot saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not save confusion matrix plot: %s", exc)


def plot_feature_importance(
    model: object,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of feature importances.

    Args:
        model: Fitted model with feature_importances_ attribute (or coef_).
        feature_names: List of feature names matching model input.
        output_path: Path to save the PNG file.
        top_n: Number of top features to display.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model has no feature importance attribute -- skipping plot")
            return

        indices = np.argsort(importances)[::-1][:top_n]
        top_names = [feature_names[i] for i in indices]
        top_values = importances[indices]

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
        ax.barh(range(len(top_names)), top_values[::-1])
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        logger.info("Feature importance plot saved to %s", output_path)
    except Exception as exc:
        logger.warning("Could not save feature importance plot: %s", exc)


def extract_feature_importance(
    model: object,
    feature_names: list[str],
) -> list[dict]:
    """Extract ordered feature importance as a list of dicts.

    Args:
        model: Fitted classifier with feature_importances_ or coef_.
        feature_names: List of feature names.

    Returns:
        List of {'feature': str, 'importance': float} sorted descending by importance.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return []

        paired = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": name, "importance": round(imp, 6)} for name, imp in paired]
    except Exception as exc:
        logger.warning("Could not extract feature importances: %s", exc)
        return []


def save_report(metrics: dict, path: Path) -> None:
    """Write the evaluation metrics to a JSON report.

    Args:
        metrics: Dictionary of computed metrics.
        path: Output JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation report saved to %s", path)


def main() -> None:
    """Load model and test set, evaluate, save report and plots."""
    config = load_config()

    model_path = PROJECT_ROOT / config["output"]["model_dir"] / config["output"]["model_filename"]
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]

    model = load_model(model_path)

    X_test = pd.read_csv(processed_dir / "X_test.csv", encoding="utf-8")
    y_test = pd.read_csv(processed_dir / "y_test.csv", encoding="utf-8").squeeze()

    logger.info("Evaluating on test set: %d rows", len(X_test))

    metrics = evaluate(model, X_test, y_test)

    feature_names = list(X_test.columns)
    feature_importance = extract_feature_importance(model, feature_names)
    metrics["feature_importance"] = feature_importance

    report_path = metrics_dir / config["output"]["report_filename"]
    save_report(metrics, report_path)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        labels=["on_track (0)", "at_risk (1)"],
        output_path=metrics_dir / "confusion_matrix.png",
    )

    plot_feature_importance(
        model,
        feature_names=feature_names,
        output_path=metrics_dir / "feature_importance.png",
        top_n=20,
    )

    recall_threshold = config["evaluation"].get("recall_threshold", 0.75)
    if metrics["recall"] >= recall_threshold:
        logger.info(
            "SUCCESS: Recall %.4f >= threshold %.2f",
            metrics["recall"], recall_threshold,
        )
    else:
        logger.warning(
            "WARNING: Recall %.4f < threshold %.2f -- consider model improvements",
            metrics["recall"], recall_threshold,
        )


if __name__ == "__main__":
    main()
