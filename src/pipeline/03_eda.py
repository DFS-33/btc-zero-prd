#!/usr/bin/env python3
"""03_eda.py -- Exploratory Data Analysis.

Computes class distribution, missing value rates, and feature correlations
on the validated merged dataset. Saves results to metrics/eda_summary.json.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import NUMERIC_FEATURES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    """Load pipeline configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_distribution(df: pd.DataFrame, target_col: str) -> dict:
    """Compute class counts and percentages for the binary target.

    Args:
        df: Validated DataFrame containing the binary target column.
        target_col: Name of the binary target column.

    Returns:
        Dictionary with class counts, percentages, and total rows.
    """
    if target_col not in df.columns:
        logger.warning("Target column '%s' not found for class distribution", target_col)
        return {}

    counts = df[target_col].value_counts().to_dict()
    total = len(df)

    distribution = {
        "total_rows": total,
        "classes": {},
    }

    for cls_val, count in counts.items():
        label = "at_risk" if int(cls_val) == 1 else "on_track"
        distribution["classes"][str(cls_val)] = {
            "label": label,
            "count": int(count),
            "percentage": round(count / total * 100, 2),
        }

    logger.info(
        "Class distribution -- total: %d | %s",
        total,
        " | ".join(
            f"{v['label']}: {v['count']} ({v['percentage']}%)"
            for v in distribution["classes"].values()
        ),
    )
    return distribution


def missing_value_report(df: pd.DataFrame) -> dict:
    """Compute missing value counts and rates per column.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dictionary mapping column name to {'count': int, 'pct': float}.
    """
    total = len(df)
    report: dict = {}

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            report[col] = {
                "null_count": null_count,
                "null_pct": round(null_count / total * 100, 2),
            }

    columns_with_nulls = len(report)
    logger.info(
        "Missing values: %d/%d columns have nulls",
        columns_with_nulls, len(df.columns),
    )
    return report


def top_correlations(df: pd.DataFrame, target_col: str, n: int = 15) -> dict:
    """Compute top-N absolute correlations of numeric features with the target.

    Args:
        df: DataFrame containing numeric features and binary target.
        target_col: Column name to correlate against.
        n: Number of top correlations to return.

    Returns:
        Dictionary mapping feature name to correlation value (sorted descending).
    """
    available_features = [f for f in NUMERIC_FEATURES if f in df.columns]

    if target_col not in df.columns:
        logger.warning("Target '%s' not found -- skipping correlation analysis", target_col)
        return {}

    numeric_df = df[available_features + [target_col]].copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    corr_series = numeric_df[available_features].corrwith(numeric_df[target_col])
    corr_sorted = corr_series.abs().sort_values(ascending=False).head(n)

    result = {
        feat: round(float(corr_series[feat]), 4)
        for feat in corr_sorted.index
        if feat in corr_series.index
    }

    logger.info(
        "Top %d correlations with '%s': %s",
        n, target_col,
        ", ".join(f"{k}={v:.3f}" for k, v in list(result.items())[:5]),
    )
    return result


def pairwise_correlation_matrix(df: pd.DataFrame) -> dict:
    """Compute full pairwise correlation matrix of numeric features.

    Args:
        df: DataFrame containing feature columns.

    Returns:
        Nested dictionary: {feature: {other_feature: correlation_value}}.
    """
    available_features = [f for f in NUMERIC_FEATURES if f in df.columns]
    numeric_df = df[available_features].copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    corr_matrix = numeric_df.corr()
    return {
        row: {col: round(float(val), 4) for col, val in corr_matrix[row].items()}
        for row in corr_matrix.index
    }


def save_eda_report(results: dict, output_dir: Path) -> None:
    """Save the EDA summary to a JSON file.

    Args:
        results: Combined EDA results dictionary.
        output_dir: Directory where eda_summary.json will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eda_summary.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("EDA report saved to %s", output_path)


def main() -> None:
    """Load validated.csv, run EDA, save eda_summary.json to metrics/."""
    config = load_config()

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    validated_path = processed_dir / "validated.csv"

    logger.info("Loading validated dataset from %s", validated_path)
    df = pd.read_csv(validated_path, encoding="utf-8")
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    binary_col = config["data"]["binary_target_column"]

    dist = class_distribution(df, binary_col)
    missing = missing_value_report(df)
    top_corr = top_correlations(df, binary_col, n=15)
    corr_matrix = pairwise_correlation_matrix(df)

    eda_report = {
        "class_distribution": dist,
        "missing_values": missing,
        "top_correlations_with_target": top_corr,
        "pairwise_correlation_matrix": corr_matrix,
    }

    output_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    save_eda_report(eda_report, output_dir)


if __name__ == "__main__":
    main()
