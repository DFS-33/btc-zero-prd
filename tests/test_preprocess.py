"""Unit tests for 04_preprocess.py -- imputation, encoding, scaling, and splits."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _import_pipeline_module(filename: str):
    script_path = PROJECT_ROOT / "src" / "pipeline" / filename
    spec = importlib.util.spec_from_file_location("pipeline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preprocess_module():
    return _import_pipeline_module("04_preprocess.py")


@pytest.fixture
def large_validated_df() -> pd.DataFrame:
    """Larger synthetic dataset for split ratio testing (100 rows)."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "fase": rng.integers(1, 8, n),
        "inde": rng.uniform(3, 10, n),
        "pedra": rng.choice(["Quartzo", "Ametista", "Top\u00e1zio"], n),
        "idade": rng.integers(8, 20, n),
        "genero": rng.choice(["Menina", "Menino", "Feminino", "Masculino"], n),
        "ano_ingresso": rng.integers(2015, 2023, n),
        "instituicao": rng.choice(["P\u00fablica", "Rede Decis\u00e3o"], n),
        "iaa": rng.uniform(4, 10, n),
        "ieg": rng.uniform(3, 10, n),
        "ips": rng.uniform(4, 9, n),
        "ipp": rng.uniform(5, 10, n),
        "ipv": rng.uniform(4, 10, n),
        "ida": rng.uniform(3, 10, n),
        "ian": rng.uniform(3, 10, n),
        "mat": rng.uniform(2, 10, n),
        "por": rng.uniform(2, 10, n),
        "ing": rng.uniform(2, 10, n),
        "defasagem": rng.choice([-2, -1, 0, 0, 0], n),
        "year": rng.choice([2022, 2023, 2024], n),
        "target": rng.choice([0, 1], n, p=[0.6, 0.4]),
    })


class TestEngineerFeatures:
    def test_avg_grades_column_created(self, preprocess_module, sample_validated_df):
        result = preprocess_module.engineer_features(sample_validated_df)
        assert "avg_grades" in result.columns

    def test_avg_grades_is_mean_of_mat_por_ing(self, preprocess_module):
        df = pd.DataFrame({"mat": [6.0], "por": [8.0], "ing": [10.0]})
        result = preprocess_module.engineer_features(df)
        assert abs(result["avg_grades"].iloc[0] - 8.0) < 1e-9

    def test_anos_programa_created_when_columns_available(self, preprocess_module, large_validated_df):
        result = preprocess_module.engineer_features(large_validated_df)
        assert "anos_programa" in result.columns

    def test_anos_programa_non_negative(self, preprocess_module, large_validated_df):
        result = preprocess_module.engineer_features(large_validated_df)
        assert (result["anos_programa"].dropna() >= 0).all()


class TestSelectFeatures:
    def test_target_column_removed_from_X(self, preprocess_module, large_validated_df, minimal_config):
        df = preprocess_module.engineer_features(large_validated_df)
        X, y = preprocess_module.select_features(df, minimal_config)
        assert "target" not in X.columns

    def test_y_series_has_correct_length(self, preprocess_module, large_validated_df, minimal_config):
        df = preprocess_module.engineer_features(large_validated_df)
        X, y = preprocess_module.select_features(df, minimal_config)
        assert len(y) == len(large_validated_df)

    def test_drop_columns_not_in_X(self, preprocess_module, large_validated_df, minimal_config):
        df = preprocess_module.engineer_features(large_validated_df)
        X, y = preprocess_module.select_features(df, minimal_config)
        for col in minimal_config["preprocessing"]["columns_to_drop"]:
            assert col not in X.columns


class TestBuildPreprocessor:
    def test_preprocessor_fits_and_transforms(self, preprocess_module, large_validated_df, minimal_config):
        from sklearn.compose import ColumnTransformer
        df = preprocess_module.engineer_features(large_validated_df)
        X, _ = preprocess_module.select_features(df, minimal_config)
        preprocessor = preprocess_module.build_preprocessor(X)
        assert isinstance(preprocessor, ColumnTransformer)

        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(X)

    def test_no_nulls_after_transform(self, preprocess_module, large_validated_df, minimal_config):
        df = large_validated_df.copy()
        df.loc[df.index[:5], "ipp"] = None
        df = preprocess_module.engineer_features(df)
        X, _ = preprocess_module.select_features(df, minimal_config)
        preprocessor = preprocess_module.build_preprocessor(X)
        X_transformed = preprocessor.fit_transform(X)
        assert not pd.isnull(X_transformed).any()


class TestSplitData:
    def test_split_sizes_approximately_correct(self, preprocess_module, large_validated_df, minimal_config):
        df = preprocess_module.engineer_features(large_validated_df)
        X, y = preprocess_module.select_features(df, minimal_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_module.split_data(X, y, minimal_config)

        total = len(X)
        assert abs(len(X_train) / total - 0.70) < 0.05
        assert abs(len(X_val) / total - 0.15) < 0.05
        assert abs(len(X_test) / total - 0.15) < 0.05

    def test_no_overlap_between_splits(self, preprocess_module, large_validated_df, minimal_config):
        df = preprocess_module.engineer_features(large_validated_df)
        X, y = preprocess_module.select_features(df, minimal_config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_module.split_data(X, y, minimal_config)

        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_reproducibility_with_same_seed(self, preprocess_module, large_validated_df, minimal_config):
        df = preprocess_module.engineer_features(large_validated_df)
        X, y = preprocess_module.select_features(df, minimal_config)
        splits_a = preprocess_module.split_data(X, y, minimal_config)
        splits_b = preprocess_module.split_data(X, y, minimal_config)
        pd.testing.assert_frame_equal(splits_a[0].reset_index(drop=True), splits_b[0].reset_index(drop=True))

    def test_total_rows_preserved(self, preprocess_module, large_validated_df, minimal_config):
        df = preprocess_module.engineer_features(large_validated_df)
        X, y = preprocess_module.select_features(df, minimal_config)
        X_train, X_val, X_test, *_ = preprocess_module.split_data(X, y, minimal_config)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
