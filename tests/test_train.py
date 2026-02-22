"""Unit tests for 05_train.py -- model building, Recall evaluation, and best selection."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _import_pipeline_module(filename: str):
    script_path = PROJECT_ROOT / "src" / "pipeline" / filename
    spec = importlib.util.spec_from_file_location("pipeline_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_module():
    return _import_pipeline_module("05_train.py")


@pytest.fixture
def synthetic_train_val() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return small synthetic (X_train, y_train, X_val, y_val) for smoke tests."""
    rng = np.random.default_rng(42)
    n_train, n_val = 80, 20
    features = ["inde", "iaa", "ieg", "ips", "ipp", "ipv", "ida", "ian", "mat", "por", "ing", "fase", "idade"]

    X_train = pd.DataFrame(rng.uniform(3, 10, (n_train, len(features))), columns=features)
    X_val = pd.DataFrame(rng.uniform(3, 10, (n_val, len(features))), columns=features)

    y_train = pd.Series(rng.choice([0, 1], n_train, p=[0.6, 0.4]), name="target")
    y_val = pd.Series(rng.choice([0, 1], n_val, p=[0.6, 0.4]), name="target")

    return X_train, y_train, X_val, y_val


class TestBuildModels:
    def test_four_models_returned(self, train_module, minimal_config):
        models = train_module.build_models(minimal_config)
        assert len(models) == 4

    def test_expected_model_names_present(self, train_module, minimal_config):
        models = train_module.build_models(minimal_config)
        assert "RandomForest" in models
        assert "XGBoost" in models
        assert "LightGBM" in models
        assert "LogisticRegression" in models

    def test_all_models_have_fit_method(self, train_module, minimal_config):
        models = train_module.build_models(minimal_config)
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} has no fit() method"

    def test_all_models_have_predict_method(self, train_module, minimal_config):
        models = train_module.build_models(minimal_config)
        for name, model in models.items():
            assert hasattr(model, "predict"), f"{name} has no predict() method"


class TestTrainModel:
    def test_model_fits_without_error(self, train_module, synthetic_train_val):
        X_train, y_train, X_val, y_val = synthetic_train_val
        model = DummyClassifier(strategy="most_frequent")
        fitted = train_module.train_model("Dummy", model, X_train, y_train)
        assert hasattr(fitted, "predict")

    def test_fitted_model_can_predict(self, train_module, synthetic_train_val):
        X_train, y_train, X_val, y_val = synthetic_train_val
        model = DummyClassifier(strategy="most_frequent")
        fitted = train_module.train_model("Dummy", model, X_train, y_train)
        preds = fitted.predict(X_val)
        assert len(preds) == len(X_val)


class TestEvaluateRecall:
    def test_recall_is_float_in_0_1(self, train_module, synthetic_train_val):
        X_train, y_train, X_val, y_val = synthetic_train_val
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        recall = train_module.evaluate_recall(model, X_val, y_val)
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    def test_perfect_recall_returns_1(self, train_module):
        perfect_model = MagicMock()
        y_val = pd.Series([0, 1, 1, 0, 1])
        perfect_model.predict.return_value = np.array([0, 1, 1, 0, 1])
        X_val = pd.DataFrame({"f": [0] * 5})
        recall = train_module.evaluate_recall(perfect_model, X_val, y_val)
        assert recall == 1.0

    def test_zero_recall_when_no_positives_predicted(self, train_module):
        zero_model = MagicMock()
        y_val = pd.Series([0, 1, 1, 1])
        zero_model.predict.return_value = np.array([0, 0, 0, 0])
        X_val = pd.DataFrame({"f": [0] * 4})
        recall = train_module.evaluate_recall(zero_model, X_val, y_val)
        assert recall == 0.0


class TestSelectBest:
    def test_model_with_highest_recall_selected(self, train_module):
        mock_model_a = MagicMock()
        mock_model_b = MagicMock()
        mock_model_c = MagicMock()

        results = {
            "ModelA": {"model": mock_model_a, "val_recall": 0.60},
            "ModelB": {"model": mock_model_b, "val_recall": 0.85},
            "ModelC": {"model": mock_model_c, "val_recall": 0.72},
        }

        best_name, best_model, best_recall = train_module.select_best(results)
        assert best_name == "ModelB"
        assert best_recall == 0.85
        assert best_model is mock_model_b

    def test_single_model_is_always_best(self, train_module):
        mock_model = MagicMock()
        results = {"OnlyModel": {"model": mock_model, "val_recall": 0.55}}
        best_name, best_model, best_recall = train_module.select_best(results)
        assert best_name == "OnlyModel"

    def test_tie_returns_consistent_winner(self, train_module):
        mock_a = MagicMock()
        mock_b = MagicMock()
        results = {
            "A": {"model": mock_a, "val_recall": 0.80},
            "B": {"model": mock_b, "val_recall": 0.80},
        }
        best_name, _, _ = train_module.select_best(results)
        assert best_name in ["A", "B"]


class TestFullTrainingSmoke:
    def test_random_forest_trains_on_synthetic_data(self, train_module, minimal_config, synthetic_train_val):
        X_train, y_train, X_val, y_val = synthetic_train_val
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        fitted = train_module.train_model("RF", model, X_train, y_train)
        recall = train_module.evaluate_recall(fitted, X_val, y_val)
        assert 0.0 <= recall <= 1.0

    def test_predictions_have_correct_shape(self, train_module, synthetic_train_val):
        X_train, y_train, X_val, y_val = synthetic_train_val
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        fitted = train_module.train_model("RF", model, X_train, y_train)
        preds = fitted.predict(X_val)
        assert len(preds) == len(X_val)
