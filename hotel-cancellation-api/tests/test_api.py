from tests.conftest import SAMPLE_BOOKING


def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_has_required_fields(client):
    data = client.get("/health").json()
    assert {"status", "model_loaded", "model_version"} <= data.keys()


def test_health_status_is_ok_when_model_loaded(client):
    assert client.get("/health").json()["status"] == "ok"


def test_health_model_loaded_is_true(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_valid_payload_returns_200(client):
    assert client.post("/predict", json=SAMPLE_BOOKING).status_code == 200


def test_predict_response_has_required_fields(client):
    data = client.post("/predict", json=SAMPLE_BOOKING).json()
    assert {"is_cancelled", "cancellation_probability", "model_version"} <= data.keys()


def test_predict_is_cancelled_is_bool(client):
    data = client.post("/predict", json=SAMPLE_BOOKING).json()
    assert isinstance(data["is_cancelled"], bool)


def test_predict_probability_is_float_in_range(client):
    data = client.post("/predict", json=SAMPLE_BOOKING).json()
    p = data["cancellation_probability"]
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_predict_missing_required_field_returns_422(client):
    payload = {k: v for k, v in SAMPLE_BOOKING.items() if k != "hotel"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_negative_lead_time_returns_422(client):
    assert client.post("/predict", json={**SAMPLE_BOOKING, "lead_time": -1}).status_code == 422


def test_predict_invalid_hotel_returns_422(client):
    assert client.post("/predict", json={**SAMPLE_BOOKING, "hotel": "Hostel"}).status_code == 422


def test_predict_lowercase_month_returns_422(client):
    payload = {**SAMPLE_BOOKING, "arrival_date_month": "august"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_adr_above_max_returns_422(client):
    assert client.post("/predict", json={**SAMPLE_BOOKING, "adr": 9999.0}).status_code == 422


def test_metrics_returns_200(client):
    assert client.get("/metrics").status_code == 200


def test_metrics_has_required_fields(client):
    data = client.get("/metrics").json()
    assert {
        "total_predictions",
        "cancellations_predicted",
        "non_cancellations_predicted",
        "uptime_seconds",
    } <= data.keys()


def test_metrics_increments_after_predict(client):
    before = client.get("/metrics").json()["total_predictions"]
    client.post("/predict", json=SAMPLE_BOOKING)
    after = client.get("/metrics").json()["total_predictions"]
    assert after == before + 1


def test_metrics_uptime_is_positive_float(client):
    uptime = client.get("/metrics").json()["uptime_seconds"]
    assert isinstance(uptime, float)
    assert uptime >= 0.0