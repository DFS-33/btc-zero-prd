"""FastAPI application for Passos Mágicos student lag prediction."""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.observer import PredictionObserver
from src.api.predictor import Predictor
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictionResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_PATH       = PROJECT_ROOT / "model" / "best_model.pkl"
_PREPROCESSOR_PATH = PROJECT_ROOT / "model" / "preprocessor.pkl"
_SUMMARY_PATH     = PROJECT_ROOT / "metrics" / "training_summary.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model artifacts")
    app.state.predictor = Predictor(
        model_path=_MODEL_PATH,
        preprocessor_path=_PREPROCESSOR_PATH,
        summary_path=_SUMMARY_PATH,
    )
    app.state.observer = PredictionObserver()
    logger.info("Startup complete")
    yield
    app.state.observer.shutdown()
    logger.info("Shutting down")


app = FastAPI(
    title="Passos Mágicos — Student Lag Predictor",
    version="1.0.0",
    description=(
        "Predicts educational lag (defasagem) for Passos Mágicos students. "
        "POST /predict with 8 student features to receive a risk classification. "
        "Model: XGBoost (Recall=0.8893, ROC-AUC=0.9151 on held-out test set)."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health(request: Request) -> HealthResponse:
    """Return API health status and loaded model name."""
    return HealthResponse(
        status="ok",
        model=request.app.state.predictor.model_name,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info(request: Request) -> ModelInfoResponse:
    """Return model metadata from the training summary."""
    summary = request.app.state.predictor.summary
    return ModelInfoResponse(
        best_model=summary["best_model"],
        best_val_recall=summary["best_val_recall"],
        model_comparison=summary.get("model_comparison", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(body: PredictRequest, request: Request) -> PredictionResponse:
    """Predict educational lag risk for a single student.

    Returns prediction (0=on_track, 1=at_risk), probability, and label.
    Feature engineering (avg_grades, anos_programa) is applied internally.
    """
    raw = body.model_dump()

    t0 = time.perf_counter()
    result = request.app.state.predictor.predict(raw)
    latency_ms = (time.perf_counter() - t0) * 1000

    request.app.state.observer.log_prediction(
        raw_inputs={k: raw[k] for k in ("mat", "por", "ing", "fase", "idade",
                                        "genero", "instituicao", "pedra")},
        engineered={
            "avg_grades": result["avg_grades"],
            "anos_programa": result["anos_programa"],
        },
        result=result,
        latency_ms=latency_ms,
        model_name=request.app.state.predictor.model_name,
    )
    return PredictionResponse(**result)
