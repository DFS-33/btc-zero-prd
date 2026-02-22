"""API tests for the FastAPI prediction endpoint.

Requires model/best_model.pkl, model/preprocessor.pkl, and
metrics/training_summary.json to be present (run the pipeline first).
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


VALID_PAYLOAD = {
    "mat": 7.5,
    "por": 6.0,
    "ing": 5.5,
    "fase": 3,
    "idade": 14,
    "genero": "M",
    "instituicao": "Escola Municipal A",
    "pedra": "Ametista",
}

HIGH_SCORE_PAYLOAD = {
    "mat": 9.5,
    "por": 9.0,
    "ing": 8.5,
    "fase": 6,
    "idade": 17,
    "genero": "F",
    "instituicao": "Escola Municipal B",
    "pedra": "Rubi",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_model_name_present(self, client):
        data = client.get("/health").json()
        assert "model" in data
        assert len(data["model"]) > 0

    def test_health_version_present(self, client):
        data = client.get("/health").json()
        assert data["version"] == "1.0.0"


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_best_model(self, client):
        data = client.get("/model/info").json()
        assert "best_model" in data
        assert isinstance(data["best_model"], str)

    def test_model_info_has_recall(self, client):
        data = client.get("/model/info").json()
        assert "best_val_recall" in data
        assert 0.0 <= data["best_val_recall"] <= 1.0

    def test_model_info_has_comparison(self, client):
        data = client.get("/model/info").json()
        assert "model_comparison" in data
        assert isinstance(data["model_comparison"], dict)


class TestPredict:
    def test_predict_valid_returns_200(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_returns_binary_prediction(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["prediction"] in {0, 1}

    def test_predict_probability_in_range(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_label_valid(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["label"] in {"at_risk", "on_track"}

    def test_predict_label_consistent_with_prediction(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        if data["prediction"] == 1:
            assert data["label"] == "at_risk"
        else:
            assert data["label"] == "on_track"


class TestValidation:
    def test_missing_mat_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "mat"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_genero_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "genero"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_grade_below_zero_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "mat": -1.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_grade_above_ten_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "ing": 11.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_fase_below_range_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "fase": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_fase_above_range_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "fase": 9}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_idade_below_range_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "idade": 5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_idade_above_range_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "idade": 100}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_type_fase_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "fase": "abc"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422
