import pytest
from fastapi.testclient import TestClient

SAMPLE_BOOKING = {
    "hotel": "City Hotel",
    "lead_time": 45,
    "arrival_date_month": "August",
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "reserved_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 120.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
    "previous_cancellations": 0,
}

MOCK_PREDICTION = {
    "is_cancelled": True,
    "cancellation_probability": 0.7831,
    "model_version": "1.0.0",
}


@pytest.fixture
def sample_booking():
    return SAMPLE_BOOKING.copy()


@pytest.fixture
def client():
    from unittest.mock import MagicMock, patch

    from app.main import _metrics, app

    mock_predictor = MagicMock()
    mock_predictor.is_loaded = True
    mock_predictor.model_version = "1.0.0"
    mock_predictor.predict.return_value = MOCK_PREDICTION.copy()

    mock_metrics = _metrics
    mock_metrics._total = 0
    mock_metrics._cancelled = 0
    mock_metrics._not_cancelled = 0

    with patch("app.main.predictor", mock_predictor):
        yield TestClient(app)