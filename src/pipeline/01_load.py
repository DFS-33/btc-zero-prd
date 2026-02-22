#!/usr/bin/env python3
"""01_load.py -- Load, normalize, and merge three annual CSV datasets.

Reads pede_2022.csv, pede_2023.csv, and pede_2024.csv from data/raw/,
normalizes column names to a unified 24-column schema, deduplicates
columns, and outputs a single merged DataFrame to data/processed/merged.csv.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import COLUMN_MAP, DUPLICATE_COLS, EVALUATOR_COLS_PATTERN
from src.utils.logger import get_logger

logger = get_logger(__name__)

_YEAR_FILE_MAP: dict[str, int] = {
    "pede_2022.csv": 2022,
    "pede_2023.csv": 2023,
    "pede_2024.csv": 2024,
}


def load_config(config_path: Path | None = None) -> dict:
    """Load pipeline configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_single_csv(filepath: Path, sep: str = ",") -> pd.DataFrame:
    """Load a single CSV file into a DataFrame.

    Args:
        filepath: Absolute path to the CSV file.
        sep: Column separator character.

    Returns:
        Raw DataFrame as loaded from disk.
    """
    df = pd.read_csv(filepath, sep=sep, encoding="utf-8")
    logger.info("Loaded %s: %d rows x %d columns", filepath.name, len(df), len(df.columns))
    return df


def drop_duplicate_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Remove duplicate column occurrences present in specific year files.

    Strategy: keep only the first occurrence of each duplicate column name.

    Args:
        df: Raw DataFrame for the given year.
        year: Source year (2022, 2023, or 2024).

    Returns:
        DataFrame with duplicate columns removed.
    """
    if year not in DUPLICATE_COLS:
        return df

    for col_name in DUPLICATE_COLS[year]:
        positions = [i for i, c in enumerate(df.columns) if c == col_name]
        if len(positions) > 1:
            logger.warning(
                "Year %d: found %d occurrences of '%s' -- dropping all but first",
                year, len(positions), col_name,
            )
            keep_mask = [i for i in range(len(df.columns)) if i not in positions[1:]]
            df = df.iloc[:, keep_mask]

    return df


def drop_evaluator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop evaluator metadata and historical columns not used in modeling.

    Args:
        df: DataFrame with raw column names.

    Returns:
        DataFrame with evaluator/metadata columns removed.
    """
    cols_to_drop = [c for c in df.columns if c in EVALUATOR_COLS_PATTERN]
    if cols_to_drop:
        logger.info("Dropping %d evaluator/metadata columns", len(cols_to_drop))
        df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def normalize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Rename year-specific column names to the unified 24-column schema.

    Only columns present in COLUMN_MAP for the given year are kept.
    Columns not in the map are silently dropped.

    Args:
        df: DataFrame after evaluator columns have been removed.
        year: Source year to select the correct mapping.

    Returns:
        DataFrame with unified column names.
    """
    mapping = COLUMN_MAP[year]
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    unified_targets = list(mapping.values())
    available = [c for c in df.columns if c in unified_targets]
    df = df[available]

    logger.info(
        "Year %d: renamed %d columns -> kept %d unified columns",
        year, len(rename_dict), len(available),
    )
    return df


def merge_years(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-year DataFrames into a single unified DataFrame.

    Args:
        dfs: List of normalized DataFrames (one per year, already tagged with 'year').

    Returns:
        Concatenated DataFrame with reset index.
    """
    merged = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Merged %d annual DataFrames into %d rows x %d columns",
        len(dfs), len(merged), len(merged.columns),
    )
    return merged


def load_all(config: dict) -> pd.DataFrame:
    """Orchestrate loading, normalizing, and merging all CSV files.

    Args:
        config: Pipeline configuration dictionary from config.yaml.

    Returns:
        Unified merged DataFrame ready for downstream validation.
    """
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    sep = config["data"].get("sep", ",")
    files = config["data"]["files"]

    dfs: list[pd.DataFrame] = []

    for csv_file in files:
        year = _YEAR_FILE_MAP.get(csv_file)
        if year is None:
            logger.warning("Unknown year for file '%s' -- skipping", csv_file)
            continue

        filepath = raw_dir / csv_file
        df = load_single_csv(filepath, sep=sep)
        df = drop_duplicate_columns(df, year)
        df = drop_evaluator_columns(df)
        df = normalize_columns(df, year)
        df[config["data"]["year_column"]] = year

        dfs.append(df)
        logger.info("Year %d processed: %d rows retained", year, len(df))

    return merge_years(dfs)


def main() -> None:
    """Load config, run full load pipeline, save merged CSV to data/processed/."""
    config = load_config()

    merged = load_all(config)

    output_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "merged.csv"

    merged.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved merged dataset: %s (%d rows, %d columns)", output_path, len(merged), len(merged.columns))


if __name__ == "__main__":
    main()
