"""Model loading and inference for the prediction API."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Loads model artifacts at startup and runs inference for a single student.

    Inference pipeline:
      1. Engineer avg_grades and anos_programa from raw inputs
      2. Apply fitted ColumnTransformer (model/preprocessor.pkl)
      3. Classify with XGBClassifier (model/best_model.pkl)

    Note: anos_programa uses fase - 1 as a proxy for year - ano_ingresso,
    since year and ano_ingresso are batch-pipeline columns not available at
    inference time. This is a documented approximation (see DESIGN_FASTAPI.md,
    Decision 2).
    """

    def __init__(
        self,
        model_path: Path,
        preprocessor_path: Path,
        summary_path: Path,
    ) -> None:
        logger.info("Loading model from %s", model_path)
        self.model = joblib.load(model_path)

        logger.info("Loading preprocessor from %s", preprocessor_path)
        self.preprocessor = joblib.load(preprocessor_path)

        logger.info("Loading training summary from %s", summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            self.summary = json.load(f)

        self.model_name: str = self.summary.get("best_model", "Unknown")
        logger.info("Predictor ready — model: %s", self.model_name)

    def predict(self, request: dict) -> dict:
        """Run inference for a single student.

        Args:
            request: Dict with keys: mat, por, ing, fase, idade,
                     genero, instituicao, pedra.

        Returns:
            Dict with keys: prediction (int), probability (float), label (str).
        """
        mat = float(request["mat"])
        por = float(request["por"])
        ing = float(request["ing"])
        fase = int(request["fase"])

        avg_grades = float(np.mean([mat, por, ing]))
        anos_programa = max(0, fase - 1)

        X = pd.DataFrame([{
            "mat": mat,
            "por": por,
            "ing": ing,
            "fase": fase,
            "idade": int(request["idade"]),
            "genero": str(request["genero"]),
            "instituicao": str(request["instituicao"]),
            "pedra": str(request["pedra"]),
            "avg_grades": avg_grades,
            "anos_programa": anos_programa,
        }])

        X_proc = self.preprocessor.transform(X)
        prediction = int(self.model.predict(X_proc)[0])
        probability = round(float(self.model.predict_proba(X_proc)[0][1]), 4)
        label = "at_risk" if prediction == 1 else "on_track"

        logger.info(
            "Prediction: %s (prob=%.4f) for fase=%d, idade=%d",
            label, probability, fase, int(request["idade"]),
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "label": label,
        }
