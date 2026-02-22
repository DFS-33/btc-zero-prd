"""Unit tests for 02_validate.py -- target column validation and binary derivation."""

import importlib.util
import sys
from pathlib import Path

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
def validate_module():
    return _import_pipeline_module("02_validate.py")


class TestCheckTargetColumn:
    def test_existing_column_passes_exists_check(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, -1, -2]})
        results = validate_module.check_target_column(df, "defasagem")
        assert results["exists"] is True

    def test_missing_column_fails_exists_check(self, validate_module):
        df = pd.DataFrame({"other_col": [1, 2]})
        results = validate_module.check_target_column(df, "defasagem")
        assert results["exists"] is False
        assert results["all_checks_passed"] is False

    def test_no_nulls_passes_null_check(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, -1, -2, 1]})
        results = validate_module.check_target_column(df, "defasagem")
        assert results["null_count"] == 0
        assert results["null_check_passed"] is True

    def test_nulls_detected(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, None, -1]})
        results = validate_module.check_target_column(df, "defasagem")
        assert results["null_count"] == 1
        assert results["null_check_passed"] is False

    def test_numeric_column_passes_numeric_check(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, -1, -2]})
        results = validate_module.check_target_column(df, "defasagem")
        assert results["is_numeric"] is True

    def test_class_distribution_computed(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, -1, -2, 0]})
        results = validate_module.check_target_column(df, "defasagem")
        assert "at_risk_count" in results
        assert results["at_risk_count"] == 2
        assert results["on_track_count"] == 2

    def test_all_at_risk(self, validate_module):
        df = pd.DataFrame({"defasagem": [-1, -2, -3]})
        results = validate_module.check_target_column(df, "defasagem")
        assert results["at_risk_count"] == 3
        assert results["on_track_count"] == 0
        assert results["at_risk_pct"] == 100.0

    def test_all_on_track(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, 0, 1]})
        results = validate_module.check_target_column(df, "defasagem")
        assert results["on_track_count"] == 3
        assert results["at_risk_count"] == 0


class TestDeriveBinaryTarget:
    def test_negative_defasagem_maps_to_1(self, validate_module):
        df = pd.DataFrame({"defasagem": [-1, -2, 0, 1]})
        result = validate_module.derive_binary_target(df, "defasagem", "target")
        assert result.loc[0, "target"] == 1
        assert result.loc[1, "target"] == 1

    def test_zero_defasagem_maps_to_0(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, -1]})
        result = validate_module.derive_binary_target(df, "defasagem", "target")
        assert result.loc[0, "target"] == 0

    def test_positive_defasagem_maps_to_0(self, validate_module):
        df = pd.DataFrame({"defasagem": [1, 2]})
        result = validate_module.derive_binary_target(df, "defasagem", "target")
        assert (result["target"] == 0).all()

    def test_binary_column_added(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, -1, -2]})
        result = validate_module.derive_binary_target(df, "defasagem", "target")
        assert "target" in result.columns

    def test_original_column_preserved(self, validate_module):
        df = pd.DataFrame({"defasagem": [-1, 0]})
        result = validate_module.derive_binary_target(df, "defasagem", "target")
        assert "defasagem" in result.columns


class TestDropNullTargets:
    def test_rows_with_null_target_dropped(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, None, -1, None, -2]})
        result = validate_module.drop_null_targets(df, "defasagem")
        assert len(result) == 3
        assert result["defasagem"].isna().sum() == 0

    def test_no_nulls_unchanged(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, -1, -2]})
        result = validate_module.drop_null_targets(df, "defasagem")
        assert len(result) == 3

    def test_index_reset_after_drop(self, validate_module):
        df = pd.DataFrame({"defasagem": [0, None, -1]})
        result = validate_module.drop_null_targets(df, "defasagem")
        assert list(result.index) == [0, 1]
