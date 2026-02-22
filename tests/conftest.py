"""Shared pytest fixtures for Passos Magicos ML pipeline tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_2022_df() -> pd.DataFrame:
    """Minimal 2022-style DataFrame with original column names (accented)."""
    return pd.DataFrame({
        "RA": ["RA-1", "RA-2", "RA-3"],
        "Fase": [7, 5, 3],
        "INDE 22": [5.78, 7.05, 6.20],
        "Pedra 22": ["Quartzo", "Ametista", "Quartzo"],
        "Turma": ["A", "B", "A"],
        "Nome": ["Aluno-1", "Aluno-2", "Aluno-3"],
        "Idade 22": [19, 17, 15],
        "G\u00eanero": ["Menina", "Menina", "Menino"],
        "Ano ingresso": [2016, 2017, 2018],
        "Institui\u00e7\u00e3o de ensino": ["Escola P\u00fablica", "Rede Decis\u00e3o", "Escola P\u00fablica"],
        "IAA": [8.3, 8.8, 7.5],
        "IEG": [4.1, 5.2, 6.0],
        "IPS": [5.6, 6.3, 5.0],
        "IPV": [7.278, 6.778, 5.500],
        "IDA": [4.0, 6.8, 5.0],
        "IAN": [5.0, 10.0, 7.0],
        "Matem": [2.7, 6.3, 5.0],
        "Portug": [3.5, 4.5, 6.0],
        "Ingl\u00eas": [6.0, 9.7, 7.0],
        "Defas": [-1, 0, -2],
        "Fase ideal": ["Fase 8", "Fase 7", "Fase 5"],
        "Rec Psicologia": ["Requer avalia\u00e7\u00e3o", "Sem limita\u00e7\u00f5es", "Sem limita\u00e7\u00f5es"],
    })


@pytest.fixture
def sample_2023_df() -> pd.DataFrame:
    """Minimal 2023-style DataFrame including IPP and duplicate Destaque IPV."""
    return pd.DataFrame({
        "RA": ["RA-861", "RA-862"],
        "Fase": ["ALFA", "ALFA"],
        "INDE 2023": [9.31, 8.22],
        "Pedra 2023": ["Top\u00e1zio", "Top\u00e1zio"],
        "Turma": ["ALFA A", "ALFA A"],
        "Nome Anonimizado": ["Aluno-861", "Aluno-862"],
        "Idade": [8, 9],
        "G\u00eanero": ["Feminino", "Masculino"],
        "Ano ingresso": [2023, 2023],
        "Institui\u00e7\u00e3o de ensino": ["P\u00fablica", "P\u00fablica"],
        "IAA": [9.5, 8.5],
        "IEG": [10.0, 9.1],
        "IPS": [8.13, 8.14],
        "IPP": [8.44, 7.5],
        "IPV": [8.92, 8.585],
        "IDA": [9.6, 8.9],
        "IAN": [10.0, 5.0],
        "Mat": [9.8, 8.5],
        "Por": [9.4, 9.2],
        "Ing": [9.4, 9.2],
        "Defasagem": [0, -1],
        "Fase Ideal": ["ALFA", "Fase 1"],
        "Rec Psicologia": ["", ""],
        "Destaque IPV": ["DUPLICATE", "DUPLICATE"],
    })


@pytest.fixture
def sample_validated_df() -> pd.DataFrame:
    """Normalized DataFrame with unified column names and binary target."""
    return pd.DataFrame({
        "fase": [7, 5, 3, 1],
        "inde": [5.78, 7.05, 6.20, 8.00],
        "pedra": ["Quartzo", "Ametista", "Quartzo", "Top\u00e1zio"],
        "idade": [19, 17, 15, 10],
        "genero": ["Menina", "Menina", "Menino", "Masculino"],
        "ano_ingresso": [2016, 2017, 2018, 2023],
        "instituicao": ["Escola P\u00fablica", "Rede Decis\u00e3o", "Escola P\u00fablica", "P\u00fablica"],
        "iaa": [8.3, 8.8, 7.5, 9.5],
        "ieg": [4.1, 5.2, 6.0, 10.0],
        "ips": [5.6, 6.3, 5.0, 8.1],
        "ipp": [None, None, None, 8.44],
        "ipv": [7.278, 6.778, 5.500, 8.92],
        "ida": [4.0, 6.8, 5.0, 9.6],
        "ian": [5.0, 10.0, 7.0, 10.0],
        "mat": [2.7, 6.3, 5.0, 9.8],
        "por": [3.5, 4.5, 6.0, 9.4],
        "ing": [6.0, 9.7, 7.0, 9.4],
        "defasagem": [-1, 0, -2, 0],
        "year": [2022, 2022, 2022, 2023],
        "target": [1, 0, 1, 0],
    })


@pytest.fixture
def minimal_config() -> dict:
    """Minimal pipeline configuration for testing (no filesystem access needed)."""
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "files": ["pede_2022.csv", "pede_2023.csv", "pede_2024.csv"],
            "sep": ",",
            "encoding": "utf-8",
            "target_column": "defasagem",
            "binary_target_column": "target",
            "year_column": "year",
        },
        "split": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
            "stratify": True,
        },
        "preprocessing": {
            "numeric_impute_strategy": "median",
            "categorical_impute_strategy": "most_frequent",
            "numeric_features": [
                "inde", "iaa", "ieg", "ips", "ipp", "ipv",
                "ida", "ian", "mat", "por", "ing", "fase", "idade",
            ],
            "categorical_features": ["genero", "instituicao", "pedra"],
            "columns_to_drop": [
                "nome", "ra", "turma", "rec_psicologia", "fase_ideal",
                "ano_ingresso", "ativo", "escola", "defasagem", "year",
            ],
        },
        "training": {
            "random_state": 42,
            "cv_folds": 3,
            "primary_metric": "recall",
            "models": {
                "RandomForest": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "class_weight": "balanced",
                    "random_state": 42,
                },
                "XGBoost": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "random_state": 42,
                },
                "LightGBM": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "random_state": 42,
                },
                "LogisticRegression": {
                    "max_iter": 100,
                    "class_weight": "balanced",
                    "random_state": 42,
                },
            },
        },
        "output": {
            "model_dir": "model",
            "model_filename": "best_model.pkl",
            "metrics_dir": "metrics",
            "report_filename": "report.json",
            "training_summary_filename": "training_summary.json",
        },
    }
