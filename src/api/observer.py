"""Langfuse prediction observer for the Passos Mágicos API."""

import logging
import os

logger = logging.getLogger(__name__)


class PredictionObserver:
    """Logs every /predict call to Langfuse Cloud as a structured trace.

    Operates in no-op mode when LANGFUSE_PUBLIC_KEY is absent or when
    a Langfuse error occurs — prediction is never blocked by observability.

    Args:
        langfuse_client: Inject a pre-built Langfuse client (used in tests).
                         If None, auto-creates from env vars or enters no-op.
    """

    def __init__(self, langfuse_client=None) -> None:
        self._client = langfuse_client
        if self._client is None and os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                from langfuse import Langfuse
                self._client = Langfuse()
                logger.info("PredictionObserver: Langfuse tracing enabled")
            except Exception as exc:
                logger.warning("PredictionObserver: Langfuse init failed (%s) — no-op mode", exc)
        else:
            logger.debug("PredictionObserver: LANGFUSE_PUBLIC_KEY not set — no-op mode")

    def log_prediction(
        self,
        raw_inputs: dict,
        engineered: dict,
        result: dict,
        latency_ms: float,
        model_name: str = "XGBoost",
    ) -> None:
        """Log one prediction as a Langfuse trace. Silent no-op on any failure."""
        if self._client is None:
            return
        try:
            span = self._client.start_span(
                name="passos-magicos/predict",
                input={"raw": raw_inputs, "engineered": engineered},
            )
            span.update(
                output={
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "label": result["label"],
                },
                metadata={"model": model_name, "latency_ms": round(latency_ms, 2)},
            )
            span.end()
        except Exception as exc:
            logger.warning("PredictionObserver: log_prediction failed: %s", exc)

    def shutdown(self) -> None:
        """Flush pending traces on app shutdown."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
