#!/usr/bin/env python3
"""04_preprocess.py -- Impute, encode, scale, engineer features, stratified split.

Reads data/processed/validated.csv, applies the preprocessing pipeline
(median imputation, ordinal encoding, standard scaling), engineers derived
features, and produces stratified 70/15/15 train/val/test splits saved
to data/processed/ as CSV files.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    """Load pipeline configuration from config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the DataFrame.

    Derived features:
    - avg_grades: mean of mat, por, ing
    - anos_programa: year - ano_ingresso (years in program)

    Args:
        df: Validated DataFrame with unified column names.

    Returns:
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    grade_cols = [c for c in ["mat", "por", "ing"] if c in df.columns]
    if grade_cols:
        numeric_grades = df[grade_cols].apply(pd.to_numeric, errors="coerce")
        df["avg_grades"] = numeric_grades.mean(axis=1)

    if "year" in df.columns and "ano_ingresso" in df.columns:
        year_num = pd.to_numeric(df["year"], errors="coerce")
        ingresso_num = pd.to_numeric(df["ano_ingresso"], errors="coerce")
        df["anos_programa"] = (year_num - ingresso_num).clip(lower=0)

    logger.info("Feature engineering complete -- new columns: avg_grades, anos_programa")
    return df


def select_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Select modeling features and separate the binary target.

    Args:
        df: Validated DataFrame (post-engineer_features).
        config: Pipeline configuration with preprocessing.columns_to_drop.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series.
    """
    drop_cols = config["preprocessing"]["columns_to_drop"]

    target_col = config["data"]["binary_target_column"]
    y = df[target_col].copy()

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    cols_to_drop_full = [c for c in cols_to_drop + [target_col] if c in df.columns]
    X = df.drop(columns=cols_to_drop_full, errors="ignore")

    numeric_cols = set(NUMERIC_FEATURES + ["avg_grades", "anos_programa"])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce") if col in numeric_cols else X[col]

    logger.info("Feature selection: %d features, %d samples", len(X.columns), len(X))
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric imputation/scaling and categorical encoding.

    Numeric pipeline: median imputation -> StandardScaler
    Categorical pipeline: most_frequent imputation -> OrdinalEncoder

    Args:
        X: Feature DataFrame (used to detect available column sets).

    Returns:
        Configured ColumnTransformer (not yet fitted).
    """
    numeric_cols = [c for c in NUMERIC_FEATURES + ["avg_grades", "anos_programa"] if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    logger.info(
        "Preprocessor built: %d numeric cols, %d categorical cols",
        len(numeric_cols), len(categorical_cols),
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 70/15/15 train/validation/test split.

    Args:
        X: Feature matrix.
        y: Target vector.
        config: Pipeline config with split ratios and random_state.

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    val_ratio = config["split"]["val_ratio"]
    test_ratio = config["split"]["test_ratio"]
    random_state = config["split"].get("random_state", RANDOM_STATE)

    combined_val_test = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=combined_val_test,
        random_state=random_state,
        stratify=y,
    )

    relative_test = test_ratio / combined_val_test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        "Split sizes -- train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit preprocessor on train set and transform all splits.

    Args:
        preprocessor: Configured but unfitted ColumnTransformer.
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.

    Returns:
        Tuple of (X_train_proc, X_val_proc, X_test_proc) as DataFrames.
    """
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    logger.info("Preprocessor applied: %d output features", len(feature_names))
    return X_train_df, X_val_df, X_test_df


def main() -> None:
    """Load validated.csv, preprocess, split, save 6 CSVs to data/processed/."""
    config = load_config()

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    validated_path = processed_dir / "validated.csv"

    logger.info("Loading validated dataset from %s", validated_path)
    df = pd.read_csv(validated_path, encoding="utf-8")
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    df = engineer_features(df)

    X, y = select_features(df, config)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    preprocessor = build_preprocessor(X_train)
    X_train_proc, X_val_proc, X_test_proc = apply_preprocessor(
        preprocessor, X_train, X_val, X_test,
    )

    model_dir = PROJECT_ROOT / config["output"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = model_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Preprocessor saved to %s", preprocessor_path)

    splits = {
        "X_train": X_train_proc,
        "X_val": X_val_proc,
        "X_test": X_test_proc,
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }

    for name, data in splits.items():
        output_path = processed_dir / f"{name}.csv"
        data.to_csv(output_path, index=False, encoding="utf-8")
        logger.info("Saved %s: %s", name, output_path)

    logger.info("Preprocessing complete -- all splits saved to %s", processed_dir)


if __name__ == "__main__":
    main()
