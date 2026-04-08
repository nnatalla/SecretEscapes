import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import BookingFeatures
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "model.joblib"
_DEFAULT_METADATA_PATH = _PROJECT_ROOT / "models" / "metadata.json"


class Predictor:
    def __init__(
        self,
        model_path: Path = _DEFAULT_MODEL_PATH,
        metadata_path: Path = _DEFAULT_METADATA_PATH,
    ) -> None:
        try:
            self._pipeline = joblib.load(model_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Model file not found at '{model_path}'. "
                "Run 'make train' to generate the model artifact before starting the API."
            ) from exc

        try:
            metadata = json.loads(Path(metadata_path).read_text())
            self.model_version: str = metadata["model_version"]
        except FileNotFoundError:
            logger.warning(
                "metadata.json not found at '%s'. Version will be reported as 'unknown'.",
                metadata_path,
            )
            self.model_version = "unknown"

        self._is_loaded = True
        logger.info("Predictor ready. Model version: %s", self.model_version)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, features: BookingFeatures) -> dict:
        row = features.model_dump()
        df = pd.DataFrame([row])[FEATURE_COLUMNS]

        prediction = int(self._pipeline.predict(df)[0])
        probability = float(self._pipeline.predict_proba(df)[0][1])

        logger.debug("Prediction: %d (p=%.4f)", prediction, probability)

        return {
            "is_cancelled": bool(prediction),
            "cancellation_probability": round(probability, 4),
            "model_version": self.model_version,
        }