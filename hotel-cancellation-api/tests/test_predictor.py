from unittest.mock import MagicMock

import numpy as np
import pytest

from app.schemas import BookingFeatures
from tests.conftest import SAMPLE_BOOKING


def _make_mock_pipeline(pred=1, proba=0.77):
    mock = MagicMock()
    mock.predict.return_value = np.array([pred])
    mock.predict_proba.return_value = np.array([[1 - proba, proba]])
    return mock


def _make_predictor(pred=1, proba=0.77):
    from app.predictor import Predictor

    p = Predictor.__new__(Predictor)
    p._pipeline = _make_mock_pipeline(pred, proba)
    p.model_version = "1.0.0"
    p._is_loaded = True
    return p


def test_predictor_raises_runtime_error_when_model_missing(tmp_path):
    from app.predictor import Predictor

    with pytest.raises(RuntimeError, match="make train"):
        Predictor(model_path=tmp_path / "missing.joblib", metadata_path=tmp_path / "missing.json")


def test_predict_returns_required_keys():
    result = _make_predictor().predict(BookingFeatures(**SAMPLE_BOOKING))
    assert set(result.keys()) == {"is_cancelled", "cancellation_probability", "model_version"}


def test_predict_is_cancelled_is_bool():
    result = _make_predictor(pred=1).predict(BookingFeatures(**SAMPLE_BOOKING))
    assert isinstance(result["is_cancelled"], bool)


def test_predict_probability_is_float():
    result = _make_predictor(proba=0.55).predict(BookingFeatures(**SAMPLE_BOOKING))
    assert isinstance(result["cancellation_probability"], float)


def test_predict_probability_rounded_to_4dp():
    result = _make_predictor(proba=0.777777).predict(BookingFeatures(**SAMPLE_BOOKING))
    assert result["cancellation_probability"] == 0.7778


def test_predict_is_cancelled_true_when_pred_1():
    result = _make_predictor(pred=1).predict(BookingFeatures(**SAMPLE_BOOKING))
    assert result["is_cancelled"] is True


def test_predict_is_cancelled_false_when_pred_0():
    result = _make_predictor(pred=0).predict(BookingFeatures(**SAMPLE_BOOKING))
    assert result["is_cancelled"] is False


def test_predict_uses_feature_columns_in_correct_order():
    from src.features import FEATURE_COLUMNS

    p = _make_predictor()
    p.predict(BookingFeatures(**SAMPLE_BOOKING))
    call_args = p._pipeline.predict.call_args[0][0]
    assert list(call_args.columns) == FEATURE_COLUMNS