"""Unit tests for PredictionObserver."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.observer import PredictionObserver
from src.api.app import app

VALID_RAW = {
    "mat": 7.5, "por": 6.0, "ing": 5.5, "fase": 3,
    "idade": 14, "genero": "M", "instituicao": "Escola A", "pedra": "Ametista",
}
ENGINEERED = {"avg_grades": 6.33, "anos_programa": 2}
RESULT = {"prediction": 0, "probability": 0.3885, "label": "on_track",
          "avg_grades": 6.33, "anos_programa": 2}


class TestPredictionObserverUnit:
    def test_log_prediction_calls_trace(self):
        mock_client = MagicMock()
        observer = PredictionObserver(langfuse_client=mock_client)
        observer.log_prediction(VALID_RAW, ENGINEERED, RESULT, 12.4)
        mock_client.trace.assert_called_once()

    def test_trace_contains_correct_name(self):
        mock_client = MagicMock()
        observer = PredictionObserver(langfuse_client=mock_client)
        observer.log_prediction(VALID_RAW, ENGINEERED, RESULT, 12.4)
        call_kwargs = mock_client.trace.call_args.kwargs
        assert call_kwargs["name"] == "passos-magicos/predict"

    def test_trace_input_has_raw_and_engineered(self):
        mock_client = MagicMock()
        observer = PredictionObserver(langfuse_client=mock_client)
        observer.log_prediction(VALID_RAW, ENGINEERED, RESULT, 12.4)
        call_kwargs = mock_client.trace.call_args.kwargs
        assert "raw" in call_kwargs["input"]
        assert "engineered" in call_kwargs["input"]

    def test_trace_metadata_has_latency(self):
        mock_client = MagicMock()
        observer = PredictionObserver(langfuse_client=mock_client)
        observer.log_prediction(VALID_RAW, ENGINEERED, RESULT, 12.4)
        call_kwargs = mock_client.trace.call_args.kwargs
        assert call_kwargs["metadata"]["latency_ms"] == 12.4

    def test_no_op_when_no_client(self):
        observer = PredictionObserver(langfuse_client=None)
        observer.log_prediction(VALID_RAW, ENGINEERED, RESULT, 5.0)

    def test_no_op_when_trace_raises(self):
        mock_client = MagicMock()
        mock_client.trace.side_effect = Exception("network error")
        observer = PredictionObserver(langfuse_client=mock_client)
        observer.log_prediction(VALID_RAW, ENGINEERED, RESULT, 5.0)

    def test_shutdown_calls_flush(self):
        mock_client = MagicMock()
        observer = PredictionObserver(langfuse_client=mock_client)
        observer.shutdown()
        mock_client.flush.assert_called_once()


class TestPredictEndpointWithObserver:
    """Verify /predict still works correctly with observer in no-op mode (no creds in CI)."""

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app) as c:
            yield c

    def test_predict_returns_200_without_langfuse_creds(self, client):
        response = client.post("/predict", json=VALID_RAW)
        assert response.status_code == 200

    def test_predict_result_unaffected_by_observer(self, client):
        data = client.post("/predict", json=VALID_RAW).json()
        assert data["prediction"] in {0, 1}
        assert 0.0 <= data["probability"] <= 1.0
        assert data["label"] in {"at_risk", "on_track"}

    def test_engineered_features_not_in_response(self, client):
        """PredictionResponse only exposes prediction/probability/label."""
        data = client.post("/predict", json=VALID_RAW).json()
        assert "avg_grades" not in data
        assert "anos_programa" not in data
