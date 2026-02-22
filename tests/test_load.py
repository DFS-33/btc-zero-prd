"""Unit tests for 01_load.py -- column normalization and merge logic."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _import_pipeline_module(filename: str):
    """Dynamically load a pipeline script by filename to allow digit-prefixed names."""
    script_path = PROJECT_ROOT / "src" / "pipeline" / filename
    spec = importlib.util.spec_from_file_location("pipeline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def load_module():
    return _import_pipeline_module("01_load.py")


class TestNormalizeColumns:
    def test_2022_columns_renamed_correctly(self, load_module, sample_2022_df):
        result = load_module.normalize_columns(sample_2022_df, year=2022)
        assert "defasagem" in result.columns
        assert "mat" in result.columns
        assert "por" in result.columns
        assert "ing" in result.columns
        assert "genero" in result.columns
        assert "instituicao" in result.columns

    def test_original_year_specific_names_removed(self, load_module, sample_2022_df):
        result = load_module.normalize_columns(sample_2022_df, year=2022)
        assert "Defas" not in result.columns
        assert "Matem" not in result.columns
        assert "Portug" not in result.columns

    def test_2022_ipp_absent_from_output(self, load_module, sample_2022_df):
        result = load_module.normalize_columns(sample_2022_df, year=2022)
        assert "ipp" not in result.columns

    def test_2023_ipp_present(self, load_module, sample_2023_df):
        clean_df = sample_2023_df.loc[:, ~sample_2023_df.columns.duplicated()].copy()
        result = load_module.normalize_columns(clean_df, year=2023)
        assert "ipp" in result.columns

    def test_unknown_columns_dropped(self, load_module, sample_2022_df):
        df_extra = sample_2022_df.copy()
        df_extra["UNKNOWN_COL"] = "garbage"
        result = load_module.normalize_columns(df_extra, year=2022)
        assert "UNKNOWN_COL" not in result.columns

    def test_only_mapped_columns_retained(self, load_module, sample_2022_df):
        result = load_module.normalize_columns(sample_2022_df, year=2022)
        from src.utils.constants import COLUMN_MAP
        expected_unified = set(COLUMN_MAP[2022].values())
        output_cols = set(result.columns)
        assert output_cols.issubset(expected_unified)


class TestDropDuplicateColumns:
    def test_year_without_duplicates_unchanged(self, load_module, sample_2022_df):
        col_count = len(sample_2022_df.columns)
        result = load_module.drop_duplicate_columns(sample_2022_df, year=2022)
        assert len(result.columns) == col_count

    def test_2023_duplicate_destaque_ipv_kept_once(self, load_module):
        df = pd.DataFrame(
            [["RA-1", "val1", "val2", 0]],
            columns=["RA", "Destaque IPV", "Destaque IPV", "Defasagem"],
        )
        result = load_module.drop_duplicate_columns(df, year=2023)
        assert result.columns.tolist().count("Destaque IPV") == 1

    def test_2024_duplicate_ativo_kept_once(self, load_module):
        df = pd.DataFrame(
            [["RA-1", "Cursando", "Cursando", 0]],
            columns=["RA", "Ativo/ Inativo", "Ativo/ Inativo", "Defasagem"],
        )
        result = load_module.drop_duplicate_columns(df, year=2024)
        assert result.columns.tolist().count("Ativo/ Inativo") == 1

    def test_first_occurrence_kept(self, load_module):
        df = pd.DataFrame(
            [["keep_val", "drop_val", 0]],
            columns=["Destaque IPV", "Destaque IPV", "Defasagem"],
        )
        result = load_module.drop_duplicate_columns(df, year=2023)
        assert result["Destaque IPV"].iloc[0] == "keep_val"


class TestDropEvaluatorColumns:
    def test_evaluator_columns_removed(self, load_module):
        df = pd.DataFrame({
            "RA": ["RA-1"],
            "Fase": [5],
            "Avaliador1": ["A1"],
            "Rec Av1": ["Mantido"],
            "Cg": [100],
            "Indicado": ["Sim"],
            "Defas": [-1],
        })
        result = load_module.drop_evaluator_columns(df)
        for col in ["Avaliador1", "Rec Av1", "Cg", "Indicado"]:
            assert col not in result.columns

    def test_core_feature_columns_preserved(self, load_module):
        df = pd.DataFrame({
            "RA": ["RA-1"],
            "Fase": [5],
            "IAA": [8.5],
            "Defas": [-1],
        })
        result = load_module.drop_evaluator_columns(df)
        assert "RA" in result.columns
        assert "Defas" in result.columns
        assert "IAA" in result.columns


class TestMergeYears:
    def test_row_count_is_sum_of_inputs(self, load_module):
        df1 = pd.DataFrame({"a": [1, 2], "year": [2022, 2022]})
        df2 = pd.DataFrame({"a": [3, 4, 5], "year": [2023, 2023, 2023]})
        result = load_module.merge_years([df1, df2])
        assert len(result) == 5

    def test_index_is_reset(self, load_module):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        result = load_module.merge_years([df1, df2])
        assert list(result.index) == [0, 1, 2, 3]

    def test_columns_from_both_dfs_present(self, load_module):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "c": [4]})
        result = load_module.merge_years([df1, df2])
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns

    def test_single_df_passthrough(self, load_module):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = load_module.merge_years([df])
        assert len(result) == 3

    def test_year_values_preserved(self, load_module):
        df1 = pd.DataFrame({"defasagem": [0, -1], "year": [2022, 2022]})
        df2 = pd.DataFrame({"defasagem": [-1], "year": [2023]})
        result = load_module.merge_years([df1, df2])
        assert set(result["year"].unique()) == {2022, 2023}
