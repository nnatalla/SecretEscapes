import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.predictor import Predictor
from app.schemas import (
    BookingFeatures,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


class _MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total: int = 0
        self._cancelled: int = 0
        self._not_cancelled: int = 0
        self._start: float = time.monotonic()

    def record(self, is_cancelled: bool) -> None:
        with self._lock:
            self._total += 1
            if is_cancelled:
                self._cancelled += 1
            else:
                self._not_cancelled += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total_predictions": self._total,
                "cancellations_predicted": self._cancelled,
                "non_cancellations_predicted": self._not_cancelled,
                "uptime_seconds": round(time.monotonic() - self._start, 2),
            }


_metrics = _MetricsCollector()
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global predictor
    try:
        predictor = Predictor()
        logger.info("Startup complete")
    except RuntimeError as exc:
        logger.critical("Model failed to load: %s", exc)
        logger.critical("API starting in DEGRADED mode")
        predictor = None

    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="Hotel Booking Cancellation API",
    description="Predicts whether a hotel booking will be cancelled using a RandomForest classifier.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"], summary="Liveness check")
async def health() -> HealthResponse:
    loaded = predictor is not None and predictor.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=predictor.model_version if loaded else "N/A",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["ml"], summary="Predict booking cancellation")
async def predict(features: BookingFeatures) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health for service status.")
    try:
        result = predictor.predict(features)
        _metrics.record(result["is_cancelled"])
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception("Unhandled error during prediction")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.") from exc


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"], summary="Request counters and uptime")
async def get_metrics() -> MetricsResponse:
    return MetricsResponse(**_metrics.snapshot())