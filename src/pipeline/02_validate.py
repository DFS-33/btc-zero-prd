#!/usr/bin/env python3
"""02_validate.py -- Validate target column integrity before training.

Reads data/processed/merged.csv, validates the 'defasagem' column for
nulls and expected integer range, derives the binary 'target' column
(1 if defasagem < 0 else 0), and saves to data/processed/validated.csv.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

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


def check_target_column(df: pd.DataFrame, target_col: str) -> dict:
    """Validate the target column for nulls, type, and expected value range.

    Args:
        df: Merged DataFrame containing the target column.
        target_col: Name of the target column to validate.

    Returns:
        Dictionary with keys: 'exists', 'null_count', 'is_numeric',
        'unique_values', 'class_distribution', 'all_checks_passed'.
    """
    results: dict = {}

    results["exists"] = target_col in df.columns
    if not results["exists"]:
        logger.error("Target column '%s' not found in DataFrame", target_col)
        results["all_checks_passed"] = False
        return results

    col = df[target_col]

    null_count = int(col.isna().sum())
    results["null_count"] = null_count
    results["null_check_passed"] = null_count == 0

    numeric_col = pd.to_numeric(col, errors="coerce")
    coerced_nulls = int(numeric_col.isna().sum())
    results["is_numeric"] = coerced_nulls == null_count
    results["coerce_loss_count"] = coerced_nulls - null_count

    unique_vals = sorted([v for v in col.dropna().unique() if pd.notna(v)])
    results["unique_values"] = [str(v) for v in unique_vals]

    value_counts = col.value_counts(dropna=True).to_dict()
    results["class_distribution"] = {str(k): int(v) for k, v in value_counts.items()}

    total_non_null = len(col.dropna())
    if total_non_null > 0:
        at_risk = int((pd.to_numeric(col, errors="coerce") < 0).sum())
        on_track = int((pd.to_numeric(col, errors="coerce") >= 0).sum())
        results["at_risk_count"] = at_risk
        results["on_track_count"] = on_track
        results["at_risk_pct"] = round(at_risk / total_non_null * 100, 2)

    results["all_checks_passed"] = (
        results["exists"]
        and results["null_check_passed"]
        and results["is_numeric"]
    )

    return results


def derive_binary_target(df: pd.DataFrame, target_col: str, binary_col: str) -> pd.DataFrame:
    """Derive binary classification target from defasagem column.

    Binary rule: target = 1 if defasagem < 0 (at risk), else 0 (on track).

    Args:
        df: DataFrame with the numeric target column.
        target_col: Name of the source defasagem column.
        binary_col: Name of the new binary column to create.

    Returns:
        DataFrame with the binary target column appended.
    """
    numeric_vals = pd.to_numeric(df[target_col], errors="coerce")
    df = df.copy()
    df[binary_col] = (numeric_vals < 0).astype(int)

    positive_count = int(df[binary_col].sum())
    total = len(df)
    logger.info(
        "Binary target derived: %d at-risk (%.1f%%), %d on-track (%.1f%%)",
        positive_count, positive_count / total * 100,
        total - positive_count, (total - positive_count) / total * 100,
    )
    return df


def drop_null_targets(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop rows with null values in the target column and log the count.

    Args:
        df: Merged DataFrame.
        target_col: Column to check for nulls.

    Returns:
        DataFrame with null-target rows removed.
    """
    null_mask = df[target_col].isna()
    null_count = null_mask.sum()
    if null_count > 0:
        logger.warning("Dropping %d rows with null '%s' values", null_count, target_col)
        df = df[~null_mask].reset_index(drop=True)
    return df


def report_validation(results: dict) -> None:
    """Log validation results with pass/fail status for each check.

    Args:
        results: Dictionary returned by check_target_column.
    """
    checks = {
        "Target column exists": results.get("exists", False),
        "No null values in target": results.get("null_check_passed", False),
        "Target is numeric": results.get("is_numeric", False),
    }

    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s", status, check_name)

    if "at_risk_pct" in results:
        logger.info(
            "Class distribution: %.1f%% at-risk, %.1f%% on-track",
            results["at_risk_pct"],
            100 - results["at_risk_pct"],
        )


def main() -> None:
    """Load merged.csv, validate target column, derive binary target, save validated.csv."""
    config = load_config()

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    merged_path = processed_dir / "merged.csv"

    logger.info("Loading merged dataset from %s", merged_path)
    df = pd.read_csv(merged_path, encoding="utf-8")
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    target_col = config["data"]["target_column"]
    binary_col = config["data"]["binary_target_column"]

    results = check_target_column(df, target_col)
    report_validation(results)

    if not results.get("exists", False):
        logger.error("Target column missing -- cannot proceed")
        sys.exit(1)

    df = drop_null_targets(df, target_col)

    re_results = check_target_column(df, target_col)
    if not re_results.get("null_check_passed", False):
        logger.error("Null check still failing after drop -- aborting")
        sys.exit(1)

    df = derive_binary_target(df, target_col, binary_col)

    output_path = processed_dir / "validated.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved validated dataset: %s (%d rows)", output_path, len(df))

    summary_path = processed_dir / "validation_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved validation report: %s", summary_path)


if __name__ == "__main__":
    main()
