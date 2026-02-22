"""Pydantic v2 request and response schemas for the prediction API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    mat: float = Field(..., ge=0.0, le=10.0, description="Mathematics grade (0–10)")
    por: float = Field(..., ge=0.0, le=10.0, description="Portuguese grade (0–10)")
    ing: float = Field(..., ge=0.0, le=10.0, description="English grade (0–10)")
    fase: int  = Field(..., ge=1,   le=8,    description="Academic phase/level (1–8)")
    idade: int = Field(..., ge=7,   le=25,   description="Student age in years (7–25)")
    genero: str     = Field(..., description="Gender (e.g. M or F)")
    instituicao: str = Field(..., description="School institution name")
    pedra: str       = Field(..., description="Evaluation stone level (e.g. Ametista, Rubi, Topázio, Quartzo)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "mat": 7.5,
                "por": 6.0,
                "ing": 5.5,
                "fase": 3,
                "idade": 14,
                "genero": "M",
                "instituicao": "Escola Municipal A",
                "pedra": "Ametista",
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int   = Field(..., description="0 = on_track, 1 = at_risk")
    probability: float = Field(..., description="Probability of at_risk class (0.0–1.0, 4 decimal places)")
    label: str         = Field(..., description="Human-readable label: 'at_risk' or 'on_track'")


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class ModelInfoResponse(BaseModel):
    best_model: str
    best_val_recall: float
    model_comparison: dict
